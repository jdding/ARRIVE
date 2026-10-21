#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::int64_t kDay = 86400;
constexpr std::int64_t kSessionGap = 1800;
constexpr std::int64_t kMinimumHiatus = 30 * kDay;
constexpr std::size_t kHistory = 128;
constexpr std::size_t kHorizon = 64;
constexpr std::int32_t kUnknown = 1;
constexpr std::int32_t kFirstItem = 3;

struct Event {
  std::uint32_t time;
  std::uint32_t raw_track;
  bool operator<(const Event& other) const {
    return time == other.time ? raw_track < other.raw_track : time < other.time;
  }
  bool operator==(const Event& other) const {
    return time == other.time && raw_track == other.raw_track;
  }
};

std::int64_t days_from_civil(int y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<int>(doe) - 719468;
}

std::int64_t parse_time(std::string_view s) {
  if (s.size() < 19) throw std::runtime_error("invalid_timestamp");
  const auto d2 = [&](std::size_t at) { return (s[at] - '0') * 10 + (s[at + 1] - '0'); };
  const int y = (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0');
  return days_from_civil(y, static_cast<unsigned>(d2(5)), static_cast<unsigned>(d2(8))) * kDay
       + d2(11) * 3600 + d2(14) * 60 + d2(17);
}

template <typename T>
void write_npy(const std::filesystem::path& path, const std::vector<T>& values,
               const std::string& descr, const std::string& shape) {
  std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
  const std::size_t prefix = 10;
  const std::size_t padding = (16 - ((prefix + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max()) throw std::runtime_error("npy_header_too_large");
  std::ofstream out(path, std::ios::binary);
  const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
  const std::uint16_t length = static_cast<std::uint16_t>(header.size());
  const unsigned char length_bytes[] = {static_cast<unsigned char>(length & 0xff), static_cast<unsigned char>(length >> 8)};
  out.write(reinterpret_cast<const char*>(length_bytes), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!out) throw std::runtime_error("npy_write_failed:" + path.string());
}

int bucket_of(std::int64_t gap) {
  if (gap < 90 * kDay) return 0;
  if (gap < 180 * kDay) return 1;
  if (gap <= 365 * kDay) return 2;
  return 3;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: materialize_music4all_onion_test OUTPUT_DIR TRACK_IDENTITY_TSV < decompressed.tsv\n";
    return 2;
  }
  const std::filesystem::path output(argv[1]);
  if (std::filesystem::exists(output)) {
    std::cerr << "refuse_existing_output\n";
    return 3;
  }

  std::unordered_map<std::string, std::int32_t> known_model_ids;
  {
    std::ifstream mapping(argv[2]);
    std::string line;
    if (!std::getline(mapping, line) || line != "raw_index\tmodel_id\ttrack_id\ttrain_events") {
      std::cerr << "invalid_track_identity_header\n";
      return 4;
    }
    while (std::getline(mapping, line)) {
      const std::size_t t1 = line.find('\t');
      const std::size_t t2 = line.find('\t', t1 + 1);
      const std::size_t t3 = line.rfind('\t');
      if (t1 == std::string::npos || t2 == std::string::npos || t3 <= t2) throw std::runtime_error("invalid_track_identity_row");
      const std::int32_t model_id = static_cast<std::int32_t>(std::stol(line.substr(t1 + 1, t2 - t1 - 1)));
      if (model_id < kFirstItem) throw std::runtime_error("invalid_model_id");
      known_model_ids.emplace(line.substr(t2 + 1, t3 - t2 - 1), model_id);
    }
  }

  const std::int64_t valid_start = parse_time("2005-01-01 00:00:00");
  const std::int64_t test_start = parse_time("2019-01-01 00:00:00");
  const std::int64_t test_end = parse_time("2021-01-01 00:00:00");

  std::unordered_map<std::string, std::uint32_t> raw_track_lookup;
  std::vector<std::int32_t> encoded_track;
  std::unordered_map<std::uint64_t, std::vector<Event>> user_events;
  raw_track_lookup.reserve(120000);
  encoded_track.reserve(60000);
  user_events.reserve(240000);

  auto intern_track = [&](std::string_view value) {
    auto found = raw_track_lookup.find(std::string(value));
    if (found != raw_track_lookup.end()) return found->second;
    const std::uint32_t raw = static_cast<std::uint32_t>(encoded_track.size());
    const auto known = known_model_ids.find(std::string(value));
    encoded_track.push_back(known == known_model_ids.end() ? kUnknown : known->second);
    raw_track_lookup.emplace(std::string(value), raw);
    return raw;
  };

  std::string line;
  if (!std::getline(std::cin, line) || line.rfind("user_id\ttrack_id\ttimestamp", 0) != 0) {
    std::cerr << "invalid_source_header\n";
    return 5;
  }
  std::uint64_t source_rows = 0, invalid_time_rows = 0;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::size_t t1 = line.find('\t');
    const std::size_t t2 = line.find('\t', t1 + 1);
    if (t1 == std::string::npos || t2 == std::string::npos) throw std::runtime_error("invalid_source_row");
    std::uint64_t user = 0;
    for (std::size_t i = 0; i < t1; ++i) user = user * 10 + static_cast<unsigned>(line[i] - '0');
    const std::int64_t time = parse_time(std::string_view(line).substr(t2 + 1));
    ++source_rows;
    if (time < valid_start) {
      ++invalid_time_rows;
      continue;
    }
    const std::uint32_t track = intern_track(std::string_view(line).substr(t1 + 1, t2 - t1 - 1));
    user_events[user].push_back({static_cast<std::uint32_t>(time), track});
    if (source_rows % 25000000 == 0) std::cerr << "rows=" << source_rows << "\n";
  }

  std::vector<std::uint64_t> users;
  users.reserve(user_events.size());
  for (const auto& entry : user_events) users.push_back(entry.first);
  std::sort(users.begin(), users.end());

  std::vector<std::int32_t> events;
  std::vector<std::uint32_t> event_times;
  std::vector<std::int64_t> sessions;
  std::vector<std::int64_t> label_rows;
  std::vector<std::int64_t> label_values;
  std::vector<std::int64_t> label_offsets{0};
  std::array<std::uint64_t, 4> bucket_counts{};
  std::uint64_t duplicate_rows = 0, tied_sessions = 0, right_censored = 0;

  for (std::size_t user_index = 0; user_index < users.size(); ++user_index) {
    auto& sequence = user_events.at(users[user_index]);
    std::sort(sequence.begin(), sequence.end());
    auto unique_end = std::unique(sequence.begin(), sequence.end());
    duplicate_rows += static_cast<std::uint64_t>(sequence.end() - unique_end);
    sequence.erase(unique_end, sequence.end());
    std::vector<std::size_t> starts{0};
    for (std::size_t i = 1; i < sequence.size(); ++i) {
      if (sequence[i].time - sequence[i - 1].time > kSessionGap) starts.push_back(i);
    }
    for (std::size_t session_index = 0; session_index < starts.size(); ++session_index) {
      const std::size_t start = starts[session_index];
      const std::size_t end = session_index + 1 < starts.size() ? starts[session_index + 1] : sequence.size();
      if (sequence[start].time < test_start || sequence[start].time >= test_end || start == 0) continue;
      if (end == sequence.size()) {
        ++right_censored;
        continue;
      }
      if (sequence[end - 1].time + kSessionGap > test_end) continue;
      const std::int64_t gap = sequence[start].time - sequence[start - 1].time;
      if (gap < kMinimumHiatus) continue;
      bool tied = false;
      for (std::size_t i = start + 1; i < end; ++i) tied = tied || sequence[i].time == sequence[i - 1].time;
      if (tied) {
        ++tied_sessions;
        continue;
      }
      const std::size_t suffix_stop = std::min(end, start + 1 + kHorizon);
      if (suffix_stop == start + 1) continue;

      const std::size_t context_begin = start > kHistory ? start - kHistory : 0;
      const std::int64_t offset = static_cast<std::int64_t>(events.size());
      for (std::size_t i = context_begin; i <= start; ++i) {
        events.push_back(encoded_track[sequence[i].raw_track]);
        event_times.push_back(sequence[i].time);
      }
      const std::int64_t target_start = static_cast<std::int64_t>(events.size()) - 1;
      for (std::size_t i = start + 1; i < suffix_stop; ++i) {
        events.push_back(encoded_track[sequence[i].raw_track]);
        event_times.push_back(sequence[i].time);
      }
      const std::int64_t target_end = static_cast<std::int64_t>(events.size());
      const std::int64_t row_id = static_cast<std::int64_t>(sessions.size() / 6);
      sessions.insert(sessions.end(), {
        static_cast<std::int64_t>(user_index), offset, target_start, target_end, gap, 2
      });
      label_rows.push_back(row_id);
      ++bucket_counts[bucket_of(gap)];

      std::unordered_set<std::uint32_t> seen;
      for (std::size_t i = start + 1; i < suffix_stop; ++i) {
        const std::uint32_t raw = sequence[i].raw_track;
        if (!seen.insert(raw).second) continue;
        const std::int32_t known = encoded_track[raw];
        label_values.push_back(known >= kFirstItem ? known : -static_cast<std::int64_t>(raw) - 1);
      }
      label_offsets.push_back(static_cast<std::int64_t>(label_values.size()));
    }
  }

  std::filesystem::create_directories(output);
  write_npy(output / "events.npy", events, "<i4", "(" + std::to_string(events.size()) + ",)");
  write_npy(output / "event_times.npy", event_times, "<u4", "(" + std::to_string(event_times.size()) + ",)");
  write_npy(output / "sessions.npy", sessions, "<i8", "(" + std::to_string(sessions.size() / 6) + ", 6)");
  write_npy(output / "metric_label_rows.npy", label_rows, "<i8", "(" + std::to_string(label_rows.size()) + ",)");
  write_npy(output / "metric_label_values.npy", label_values, "<i8", "(" + std::to_string(label_values.size()) + ",)");
  write_npy(output / "metric_label_offsets.npy", label_offsets, "<i8", "(" + std::to_string(label_offsets.size()) + ",)");

  std::ofstream stats(output / "TEST_STATS.json");
  stats << "{\n"
        << "  \"schema\": \"genrec.music4all_onion.test_stats.v1\",\n"
        << "  \"source_rows\": " << source_rows << ",\n"
        << "  \"invalid_timestamp_rows\": " << invalid_time_rows << ",\n"
        << "  \"users\": " << users.size() << ",\n"
        << "  \"raw_tracks\": " << encoded_track.size() << ",\n"
        << "  \"train_catalog_tracks\": " << known_model_ids.size() << ",\n"
        << "  \"exact_duplicate_rows_removed\": " << duplicate_rows << ",\n"
        << "  \"ambiguous_tie_sessions_removed\": " << tied_sessions << ",\n"
        << "  \"right_censored_sessions_removed\": " << right_censored << ",\n"
        << "  \"test_nonempty_episodes\": " << label_rows.size() << ",\n"
        << "  \"test_gap_buckets\": [" << bucket_counts[0] << ',' << bucket_counts[1] << ',' << bucket_counts[2] << ',' << bucket_counts[3] << "],\n"
        << "  \"distinct_targets\": " << label_values.size() << ",\n"
        << "  \"test_materialized\": true\n"
        << "}\n";
  std::cerr << "TEST_MATERIALIZATION_COMPLETE episodes=" << label_rows.size()
            << " targets=" << label_values.size() << " events=" << events.size() << "\n";
}
