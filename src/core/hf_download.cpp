#include "core/hf_download.h"

#include <unistd.h>  // getpid

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "core/env.h"
#include "core/logging.h"

namespace fs = std::filesystem;

namespace mlxforge {

namespace {

// curl_global_init must run once before any easy handle and is not thread-safe;
// a function-local static gives us exactly-once init plus cleanup at exit.
struct CurlGlobal {
  CurlGlobal() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobal() { curl_global_cleanup(); }
};
void ensure_curl_global() { static CurlGlobal g; }

size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
  const size_t n = size * nmemb;
  static_cast<std::string*>(userdata)->append(ptr, n);
  return n;
}

size_t write_to_ofstream(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* os = static_cast<std::ofstream*>(userdata);
  const size_t n = size * nmemb;
  os->write(ptr, static_cast<std::streamsize>(n));
  return os->good() ? n : 0;  // a short return aborts the transfer
}

// Throttled download progress: log roughly every 10% so a big safetensors pull
// shows life without flooding the log. HF's resolve URL 302-redirects to a CDN;
// curl reports a fresh dltotal for the real transfer, so reset the milestone
// tracking whenever dltotal changes (and ignore unknown/tiny totals).
struct Progress {
  std::string file;
  curl_off_t last_total = -1;
  int last_decile = -1;
};
int xferinfo(void* userdata, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
  auto* p = static_cast<Progress*>(userdata);
  // Only report on a real, sizeable body (skip redirect responses, which carry a
  // tiny or unknown content length).
  if (dltotal < 1024 * 1024) return 0;
  if (dltotal != p->last_total) {  // new transfer (e.g. after the CDN redirect)
    p->last_total = dltotal;
    p->last_decile = -1;
  }
  const int decile = static_cast<int>((10 * dlnow) / dltotal);
  if (decile > p->last_decile) {
    p->last_decile = decile;
    log::info("download {}: {}% ({:.1f}/{:.1f} MiB)", p->file, decile * 10,
              dlnow / (1024.0 * 1024.0), dltotal / (1024.0 * 1024.0));
  }
  return 0;
}

// Bearer token for gated/private repos, if the user exported one (matching the
// names huggingface_hub reads).
std::string hf_token() {
  std::string t = env_or("HF_TOKEN", "");
  if (t.empty()) t = env_or("HUGGING_FACE_HUB_TOKEN", "");
  return t;
}

// Percent-encode a path for use in a URL, leaving '/' and the RFC 3986 unreserved
// set intact (HF rfilenames are usually plain, but may contain spaces).
std::string url_encode_path(const std::string& path) {
  static const char* hex = "0123456789ABCDEF";
  std::string out;
  out.reserve(path.size());
  for (unsigned char c : path) {
    const bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
                            c == '~' || c == '/';
    if (unreserved) {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(hex[c >> 4]);
      out.push_back(hex[c & 0xF]);
    }
  }
  return out;
}

// Configure the common options shared by every request (redirects, UA, auth).
// Returns the header list the caller must curl_slist_free_all after perform.
curl_slist* common_setup(CURL* h, const std::string& url) {
  curl_easy_setopt(h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);  // HF resolve -> CDN is cross-host
  curl_easy_setopt(h, CURLOPT_MAXREDIRS, 16L);
  curl_easy_setopt(h, CURLOPT_USERAGENT, "mlxforge/0.1");
  // Keep FAILONERROR off: we read CURLINFO_RESPONSE_CODE ourselves so we can give
  // gated (401/403) vs not-found (404) specific messages.
  curl_slist* headers = nullptr;
  const std::string token = hf_token();
  if (!token.empty()) {
    headers = curl_slist_append(headers, ("Authorization: Bearer " + token).c_str());
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
  }
  return headers;
}

// Map a non-2xx HTTP status on an HF endpoint to a targeted exception.
[[noreturn]] void throw_http_error(long status, const std::string& repo_id,
                                   const std::string& what) {
  if (status == 401 || status == 403) {
    throw std::runtime_error("hf: repo '" + repo_id + "' is gated or private (" + what +
                             ", HTTP " + std::to_string(status) +
                             "); accept its terms and set HF_TOKEN");
  }
  if (status == 404) {
    throw std::runtime_error("hf: repo '" + repo_id + "' not found (" + what + ", HTTP 404)");
  }
  throw std::runtime_error("hf: " + what + " failed for '" + repo_id + "' (HTTP " +
                           std::to_string(status) + ")");
}

}  // namespace

std::vector<std::string> parse_repo_siblings(const std::string& model_info_json) {
  std::vector<std::string> files;
  const auto j = nlohmann::json::parse(model_info_json);
  const auto it = j.find("siblings");
  if (it == j.end() || !it->is_array()) return files;
  for (const auto& s : *it) {
    if (s.contains("rfilename") && s["rfilename"].is_string()) {
      files.push_back(s["rfilename"].get<std::string>());
    }
  }
  return files;
}

std::vector<std::string> select_files_to_download(const std::vector<std::string>& siblings) {
  // Exact filenames the engine (or its tokenizer) may read.
  static const std::vector<std::string> kKeep = {
      "config.json",
      "generation_config.json",
      "model.safetensors.index.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "tokenizer.model",
      "special_tokens_map.json",
      "added_tokens.json",
      "vocab.json",
      "merges.txt",
  };
  auto ends_with = [](const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
  };

  std::vector<std::string> out;
  bool has_config = false, has_safetensors = false;
  for (const auto& f : siblings) {
    const bool is_safetensors = ends_with(f, ".safetensors");
    const bool keep =
        is_safetensors || std::find(kKeep.begin(), kKeep.end(), f) != kKeep.end();
    if (!keep) continue;
    out.push_back(f);
    if (f == "config.json") has_config = true;
    if (is_safetensors) has_safetensors = true;
  }

  if (!has_config) throw std::runtime_error("hf: repo has no config.json");
  if (!has_safetensors)
    throw std::runtime_error("hf: repo has no .safetensors weights (this engine loads safetensors only)");
  return out;
}

std::string hf_download_repo(const std::string& repo_id, const std::string& dest_dir,
                             const std::string& revision) {
  ensure_curl_global();

  // 1. List the repo's files via the model-info API.
  const std::string api_url = "https://huggingface.co/api/models/" + repo_id +
                              "?revision=" + revision;
  std::string api_body;
  {
    CURL* h = curl_easy_init();
    if (!h) throw std::runtime_error("hf: failed to init curl");
    curl_slist* headers = common_setup(h, api_url);
    curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");  // allow gzip for the JSON
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &api_body);
    const CURLcode rc = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(h);
    if (rc != CURLE_OK)
      throw std::runtime_error("hf: model-info request failed: " +
                               std::string(curl_easy_strerror(rc)));
    if (status < 200 || status >= 300) throw_http_error(status, repo_id, "model-info");
  }

  std::vector<std::string> files;
  try {
    files = select_files_to_download(parse_repo_siblings(api_body));
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("hf: could not parse model-info for '" + repo_id + "': " + e.what());
  }
  log::info("hf: downloading {} files for '{}' (rev {})", files.size(), repo_id, revision);

  // 2. Download into a sibling temp dir; rename into place only on full success.
  const fs::path final_dir(dest_dir);
  const fs::path tmp_dir = final_dir.string() + ".incomplete-" + std::to_string(::getpid());
  std::error_code ec;
  fs::remove_all(tmp_dir, ec);
  fs::create_directories(tmp_dir, ec);
  if (ec) throw std::runtime_error("hf: cannot create '" + tmp_dir.string() + "': " + ec.message());

  // RAII: drop the partial temp dir unless we explicitly commit.
  struct TmpGuard {
    fs::path dir;
    bool committed = false;
    ~TmpGuard() {
      if (!committed) {
        std::error_code e;
        fs::remove_all(dir, e);
      }
    }
  } guard{tmp_dir};

  for (const auto& file : files) {
    const std::string url =
        "https://huggingface.co/" + repo_id + "/resolve/" + revision + "/" + url_encode_path(file);
    const fs::path out_path = tmp_dir / file;
    fs::create_directories(out_path.parent_path(), ec);  // nested rfilenames

    std::ofstream os(out_path, std::ios::binary | std::ios::trunc);
    if (!os) throw std::runtime_error("hf: cannot write '" + out_path.string() + "'");

    CURL* h = curl_easy_init();
    if (!h) throw std::runtime_error("hf: failed to init curl");
    curl_slist* headers = common_setup(h, url);
    Progress prog{file};
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_ofstream);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &os);
    curl_easy_setopt(h, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(h, CURLOPT_XFERINFOFUNCTION, xferinfo);
    curl_easy_setopt(h, CURLOPT_XFERINFODATA, &prog);
    const CURLcode rc = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(h);
    os.close();

    if (rc != CURLE_OK)
      throw std::runtime_error("hf: download of '" + file +
                               "' failed: " + std::string(curl_easy_strerror(rc)));
    if (status < 200 || status >= 300) throw_http_error(status, repo_id, "download of " + file);
  }

  // 3. Commit: replace any stale dest, then atomically rename temp into place.
  fs::remove_all(final_dir, ec);
  fs::rename(tmp_dir, final_dir, ec);
  if (ec) throw std::runtime_error("hf: cannot finalize '" + final_dir.string() + "': " + ec.message());
  guard.committed = true;

  log::info("hf: model '{}' ready at '{}'", repo_id, final_dir.string());
  return final_dir.string();
}

}  // namespace mlxforge
