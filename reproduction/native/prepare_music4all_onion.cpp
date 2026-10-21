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
  std::uint32_t track;
  bool operator<(const Event& other) const {
    return time == other.time ? track < other.track : time < other.time;
  }
  bool operator==(const Event& other) const {
    return time == other.time && track == other.track;
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
  if (gap < 365 * kDay) return 2;
  return 3;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: prepare_music4all_onion OUTPUT_DIR < decompressed.tsv\n";
    return 2;
  }
  const std::filesystem::path output(argv[1]);
  if (std::filesystem::exists(output)) {
    std::cerr << "refuse_existing_output\n";
    return 3;
  }
  std::filesystem::create_directories(output);

  const std::int64_t valid_start = parse_time("2005-01-01 00:00:00");
  const std::int64_t train_end = parse_time("2018-01-01 00:00:00");
  const std::int64_t validation_end = parse_time("2019-01-01 00:00:00");

  std::unordered_map<std::string, std::uint32_t> track_lookup;
  std::vector<std::string> raw_tracks;
  std::vector<std::uint64_t> train_counts;
  std::unordered_map<std::uint64_t, std::vector<Event>> user_events;
  track_lookup.reserve(120000);
  raw_tracks.reserve(60000);
  train_counts.reserve(60000);
  user_events.reserve(240000);

  auto intern_track = [&](std::string_view raw) {
    auto found = track_lookup.find(std::string(raw));
    if (found != track_lookup.end()) return found->second;
    const std::uint32_t id = static_cast<std::uint32_t>(raw_tracks.size());
    track_lookup.emplace(std::string(raw), id);
    raw_tracks.emplace_back(raw);
    train_counts.push_back(0);
    return id;
  };

  std::string line;
  if (!std::getline(std::cin, line) || line.rfind("user_id\ttrack_id\ttimestamp", 0) != 0) {
    std::cerr << "invalid_header\n";
    return 4;
  }
  std::uint64_t source_rows = 0, invalid_time_rows = 0;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::size_t t1 = line.find('\t');
    const std::size_t t2 = line.find('\t', t1 + 1);
    if (t1 == std::string::npos || t2 == std::string::npos) throw std::runtime_error("invalid_row");
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
    if (time < train_end) ++train_counts[track];
    if (source_rows % 25000000 == 0) std::cerr << "rows=" << source_rows << "\n";
  }

  std::vector<std::uint32_t> catalog(raw_tracks.size());
  std::iota(catalog.begin(), catalog.end(), 0);
  catalog.erase(std::remove_if(catalog.begin(), catalog.end(), [&](std::uint32_t id) {
    return train_counts[id] == 0;
  }), catalog.end());
  std::sort(catalog.begin(), catalog.end(), [&](std::uint32_t left, std::uint32_t right) {
    return train_counts[left] != train_counts[right]
        ? train_counts[left] > train_counts[right]
        : raw_tracks[left] < raw_tracks[right];
  });
  std::vector<std::int32_t> encoded_track(raw_tracks.size(), kUnknown);
  std::vector<std::int64_t> frequencies(catalog.size() + kFirstItem, 0);
  for (std::size_t index = 0; index < catalog.size(); ++index) {
    encoded_track[catalog[index]] = static_cast<std::int32_t>(index + kFirstItem);
    frequencies[index + kFirstItem] = static_cast<std::int64_t>(train_counts[catalog[index]]);
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
  std::array<std::uint64_t, 2> episode_counts{};
  std::array<std::uint64_t, 2> nonempty_counts{};
  std::array<std::array<std::uint64_t, 4>, 2> bucket_counts{};
  std::uint64_t duplicate_rows = 0, tied_sessions = 0;

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
      const int split = sequence[start].time < train_end ? 0 : sequence[start].time < validation_end ? 1 : -1;
      if (split < 0 || start == 0) continue;
      const std::int64_t boundary = split == 0 ? train_end : validation_end;
      if (sequence[end - 1].time + kSessionGap > boundary) continue;
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
        events.push_back(encoded_track[sequence[i].track]);
        event_times.push_back(sequence[i].time);
      }
      const std::int64_t target_start = static_cast<std::int64_t>(events.size()) - 1;
      for (std::size_t i = start + 1; i < suffix_stop; ++i) {
        events.push_back(encoded_track[sequence[i].track]);
        event_times.push_back(sequence[i].time);
      }
      const std::int64_t target_end = static_cast<std::int64_t>(events.size());
      const std::int64_t row_id = static_cast<std::int64_t>(sessions.size() / 6);
      sessions.insert(sessions.end(), {
        static_cast<std::int64_t>(user_index), offset, target_start, target_end, gap, split
      });
      ++episode_counts[split];
      ++bucket_counts[split][bucket_of(gap)];
      if (suffix_stop > start + 1) ++nonempty_counts[split];

      if (split == 1) {
        label_rows.push_back(row_id);
        std::vector<std::uint32_t> distinct;
        for (std::size_t i = start + 1; i < suffix_stop; ++i) {
          if (std::find(distinct.begin(), distinct.end(), sequence[i].track) == distinct.end()) {
            distinct.push_back(sequence[i].track);
          }
        }
        for (std::uint32_t track : distinct) {
          const std::int32_t known = encoded_track[track];
          label_values.push_back(known >= kFirstItem ? known : -static_cast<std::int64_t>(track) - 1);
        }
        label_offsets.push_back(static_cast<std::int64_t>(label_values.size()));
      }
    }
    if ((user_index + 1) % 10000 == 0) std::cerr << "users=" << (user_index + 1) << "\n";
  }

  write_npy(output / "events.npy", events, "<i4", "(" + std::to_string(events.size()) + ",)");
  write_npy(output / "event_times.npy", event_times, "<u4", "(" + std::to_string(event_times.size()) + ",)");
  write_npy(output / "sessions.npy", sessions, "<i8", "(" + std::to_string(sessions.size() / 6) + ", 6)");
  write_npy(output / "train_frequencies.npy", frequencies, "<i8", "(" + std::to_string(frequencies.size()) + ",)");
  write_npy(output / "metric_label_rows.npy", label_rows, "<i8", "(" + std::to_string(label_rows.size()) + ",)");
  write_npy(output / "metric_label_values.npy", label_values, "<i8", "(" + std::to_string(label_values.size()) + ",)");
  write_npy(output / "metric_label_offsets.npy", label_offsets, "<i8", "(" + std::to_string(label_offsets.size()) + ",)");
  std::ofstream mapping(output / "track_identity.tsv");
  mapping << "raw_index\tmodel_id\ttrack_id\ttrain_events\n";
  for (const std::uint32_t raw : catalog) {
    mapping << raw << '\t' << encoded_track[raw] << '\t' << raw_tracks[raw] << '\t' << train_counts[raw] << '\n';
  }

  std::ofstream stats(output / "PREP_STATS.json");
  stats << "{\n"
        << "  \"schema\": \"genrec.music4all_onion.prep_stats.v1\",\n"
        << "  \"source_rows\": " << source_rows << ",\n"
        << "  \"invalid_timestamp_rows\": " << invalid_time_rows << ",\n"
        << "  \"users\": " << users.size() << ",\n"
        << "  \"raw_tracks\": " << raw_tracks.size() << ",\n"
        << "  \"train_catalog_tracks\": " << catalog.size() << ",\n"
        << "  \"exact_duplicate_rows_removed\": " << duplicate_rows << ",\n"
        << "  \"ambiguous_tie_sessions_removed\": " << tied_sessions << ",\n"
        << "  \"train_episodes\": " << episode_counts[0] << ",\n"
        << "  \"train_nonempty_episodes\": " << nonempty_counts[0] << ",\n"
        << "  \"validation_episodes\": " << episode_counts[1] << ",\n"
        << "  \"validation_nonempty_episodes\": " << nonempty_counts[1] << ",\n"
        << "  \"train_gap_buckets\": [" << bucket_counts[0][0] << ',' << bucket_counts[0][1] << ',' << bucket_counts[0][2] << ',' << bucket_counts[0][3] << "],\n"
        << "  \"validation_gap_buckets\": [" << bucket_counts[1][0] << ',' << bucket_counts[1][1] << ',' << bucket_counts[1][2] << ',' << bucket_counts[1][3] << "],\n"
        << "  \"test_materialized\": false\n"
        << "}\n";
  std::cerr << "PREPARATION_COMPLETE train=" << episode_counts[0]
            << " validation=" << episode_counts[1] << " events=" << events.size() << "\n";
}
