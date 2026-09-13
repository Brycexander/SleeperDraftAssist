#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kPositions = 6;
constexpr int kQB = 0;
constexpr int kRB = 1;
constexpr int kWR = 2;
constexpr int kTE = 3;
constexpr int kK = 4;
constexpr int kDEF = 5;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct PlayerData {
    int position{};
    double value_rank{};
    double ecr{};
    double adp_mean{};
    double adp_sigma{};
    double outcome_mu{};
    double outcome_sigma{};
    double tier_gap_rank{};
    double projected_points{};
};

struct CounterRng {
    explicit CounterRng(std::uint64_t seed) : key(seed) {}

    static std::uint64_t mix(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    std::uint64_t next_u64() {
        return mix(key + counter++ * 0x9e3779b97f4a7c15ULL);
    }

    double uniform() {
        return static_cast<double>(next_u64() >> 11U) * 0x1.0p-53;
    }

    double normal() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        const double u1 = std::max(uniform(), std::numeric_limits<double>::min());
        const double u2 = uniform();
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = 2.0 * kPi * u2;
        spare = radius * std::sin(angle);
        has_spare = true;
        return radius * std::cos(angle);
    }

    std::uint64_t key;
    std::uint64_t counter{};
    double spare{};
    bool has_spare{false};
};

struct Scratch {
    Scratch(int player_count, int teams, int rounds)
        : sampled_ranks(player_count), outcomes(player_count), order(player_count),
          drafted(player_count), roster_players(teams * rounds, -1), roster_sizes(teams),
          counts(teams), base_roster_players(teams * rounds, -1),
          base_roster_sizes(teams), base_counts(teams), base_drafted(player_count) {}

    void reset() {
        drafted = base_drafted;
        roster_players = base_roster_players;
        roster_sizes = base_roster_sizes;
        counts = base_counts;
    }

    std::vector<double> sampled_ranks;
    std::vector<double> outcomes;
    std::vector<int> order;
    std::vector<std::uint8_t> drafted;
    std::vector<int> roster_players;
    std::vector<int> roster_sizes;
    std::vector<std::array<int, kPositions>> counts;
    std::vector<int> base_roster_players;
    std::vector<int> base_roster_sizes;
    std::vector<std::array<int, kPositions>> base_counts;
    std::vector<std::uint8_t> base_drafted;
};

struct RolloutSummary {
    int first_selection{-1};
    double user_score{};
    bool top_roster{};
    bool failed{};
};

struct OnClockResult {
    std::vector<int> candidate_indices;
    std::vector<double> score_sums;
    std::vector<std::uint64_t> top_counts;
    std::vector<std::uint64_t> samples;
    std::uint64_t total_rollouts{};
};

struct BeforeTurnResult {
    std::vector<std::uint64_t> selected_counts;
    std::vector<std::uint64_t> available_counts;
    std::vector<double> score_sums;
    std::vector<std::uint64_t> top_counts;
    std::uint64_t total_rollouts{};
};

class NativeDraftEngine {
public:
    explicit NativeDraftEngine(const py::dict& config) {
        teams_ = py::cast<int>(config["teams"]);
        rounds_ = py::cast<int>(config["rounds"]);
        flex_slots_ = py::cast<int>(config["flex_slots"]);
        user_roster_ = py::cast<int>(config["user_roster"]);
        next_pick_ = py::cast<int>(config["next_pick"]);
        has_projections_ = py::cast<bool>(config["has_projections"]);
        pass_td_ = py::cast<double>(config["pass_td"]);
        required_ = vector_to_array<int>(py::cast<std::vector<int>>(config["required"]));
        caps_ = vector_to_array<int>(py::cast<std::vector<int>>(config["caps"]));
        pick_owners_ = py::cast<std::vector<int>>(config["pick_owners"]);
        value_order_ = py::cast<std::vector<int>>(config["value_order"]);
        biases_ = py::cast<std::vector<std::array<double, kPositions>>>(config["biases"]);

        const auto encoded_players = py::cast<std::vector<
            std::tuple<int, double, double, double, double, double, double, double, double>>>(
                config["players"]
            );
        players_.reserve(encoded_players.size());
        for (const auto& [position, value_rank, ecr, adp_mean, adp_sigma,
                          outcome_mu, outcome_sigma, tier_gap_rank, projected_points] : encoded_players) {
            players_.push_back(PlayerData{
                position, value_rank, ecr, adp_mean, adp_sigma, outcome_mu,
                outcome_sigma, tier_gap_rank, projected_points
            });
        }
        value_ranks_.reserve(players_.size());
        for (const auto& player : players_) {
            value_ranks_.push_back(has_projections_ ? player.value_rank : player.ecr);
        }
        validate_config();
        set_state(
            py::cast<std::vector<std::vector<int>>>(config["initial_rosters"]),
            next_pick_
        );
    }

    void set_state(const std::vector<std::vector<int>>& rosters, int next_pick) {
        if (static_cast<int>(rosters.size()) != teams_) {
            throw std::invalid_argument("initial_rosters must have one entry per team");
        }
        if (next_pick < 0 || next_pick > teams_ * rounds_) {
            throw std::invalid_argument("next_pick is out of range");
        }
        next_pick_ = next_pick;
        base_roster_players_.assign(teams_ * rounds_, -1);
        base_roster_sizes_.assign(teams_, 0);
        base_counts_.assign(teams_, {});
        base_drafted_.assign(players_.size(), 0);
        for (int roster = 0; roster < teams_; ++roster) {
            if (static_cast<int>(rosters[roster].size()) > rounds_) {
                throw std::invalid_argument("a roster contains more players than draft rounds");
            }
            for (const int player : rosters[roster]) {
                if (player < 0 || player >= static_cast<int>(players_.size())) {
                    throw std::invalid_argument("roster player index is out of range");
                }
                const int offset = roster * rounds_ + base_roster_sizes_[roster]++;
                base_roster_players_[offset] = player;
                base_drafted_[player] = 1;
                const int position = players_[player].position;
                if (position >= 0 && position < kPositions) {
                    ++base_counts_[roster][position];
                }
            }
        }
    }

    std::vector<int> candidate_indices(int candidate_count) const {
        if (candidate_count < 1) {
            throw std::invalid_argument("candidate_count must be at least 1");
        }
        Scratch scratch(static_cast<int>(players_.size()), teams_, rounds_);
        load_base(scratch);
        scratch.reset();
        const int limit = std::max(40, candidate_count * 4);
        std::vector<std::pair<double, int>> scored;
        scored.reserve(limit);
        int legal_seen = 0;
        const int round = next_pick_ / teams_ + 1;
        for (const int player : value_order_) {
            if (scratch.drafted[player] ||
                !can_add(scratch, user_roster_, player)) {
                continue;
            }
            scored.emplace_back(
                pick_score(scratch, user_roster_, player, round,
                           value_ranks_[player], nullptr, next_pick_),
                player
            );
            if (++legal_seen >= limit) {
                break;
            }
        }
        std::stable_sort(scored.begin(), scored.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        if (static_cast<int>(scored.size()) > candidate_count) {
            scored.resize(candidate_count);
        }
        std::vector<int> result;
        result.reserve(scored.size());
        for (const auto& [score, player] : scored) {
            static_cast<void>(score);
            result.push_back(player);
        }
        return result;
    }

    py::dict recommend_on_clock(
        std::uint64_t total_rollouts,
        int candidate_count,
        std::uint64_t seed,
        int requested_threads
    ) const {
        if (total_rollouts < 1) {
            throw std::invalid_argument("total_rollouts must be at least 1");
        }
        const auto candidates = candidate_indices(candidate_count);
        OnClockResult result;
        {
            py::gil_scoped_release release;
            result = run_on_clock(total_rollouts, candidates, seed, requested_threads);
        }
        py::dict output;
        output["candidate_indices"] = result.candidate_indices;
        output["score_sums"] = result.score_sums;
        output["top_counts"] = result.top_counts;
        output["samples"] = result.samples;
        output["total_rollouts"] = result.total_rollouts;
        return output;
    }

    py::dict recommend_before_turn(
        std::uint64_t total_rollouts,
        std::uint64_t seed,
        int requested_threads
    ) const {
        if (total_rollouts < 1) {
            throw std::invalid_argument("total_rollouts must be at least 1");
        }
        BeforeTurnResult result;
        {
            py::gil_scoped_release release;
            result = run_before_turn(total_rollouts, seed, requested_threads);
        }
        py::dict output;
        output["selected_counts"] = result.selected_counts;
        output["available_counts"] = result.available_counts;
        output["score_sums"] = result.score_sums;
        output["top_counts"] = result.top_counts;
        output["total_rollouts"] = result.total_rollouts;
        return output;
    }

private:
    template <typename T>
    static std::array<T, kPositions> vector_to_array(const std::vector<T>& values) {
        if (values.size() != kPositions) {
            throw std::invalid_argument("position array must contain six values");
        }
        std::array<T, kPositions> result{};
        std::copy(values.begin(), values.end(), result.begin());
        return result;
    }

    void validate_config() const {
        if (teams_ < 2 || rounds_ < 1 || rounds_ > 64) {
            throw std::invalid_argument("native engine supports 2+ teams and 1-64 rounds");
        }
        if (players_.empty()) {
            throw std::invalid_argument("players cannot be empty");
        }
        if (user_roster_ < 0 || user_roster_ >= teams_) {
            throw std::invalid_argument("user_roster is out of range");
        }
        if (pick_owners_.size() != static_cast<std::size_t>(teams_ * rounds_)) {
            throw std::invalid_argument("pick_owners has the wrong length");
        }
        if (biases_.size() != static_cast<std::size_t>(teams_)) {
            throw std::invalid_argument("biases must have one entry per team");
        }
        for (const auto& player : players_) {
            if (player.position < 0 || player.position >= kPositions) {
                throw std::invalid_argument("native engine received an unsupported position");
            }
        }
    }

    void load_base(Scratch& scratch) const {
        scratch.base_roster_players = base_roster_players_;
        scratch.base_roster_sizes = base_roster_sizes_;
        scratch.base_counts = base_counts_;
        scratch.base_drafted = base_drafted_;
    }

    int resolve_threads(int requested, std::uint64_t jobs) const {
        int threads = requested;
        if (threads <= 0) {
            threads = static_cast<int>(std::thread::hardware_concurrency());
        }
        threads = std::max(1, threads);
        return static_cast<int>(std::min<std::uint64_t>(threads, jobs));
    }

    static std::uint64_t rollout_seed(std::uint64_t seed, std::uint64_t rollout) {
        return CounterRng::mix(seed ^ CounterRng::mix(rollout + 0xd1b54a32d192ed03ULL));
    }

    bool can_add(const Scratch& scratch, int roster, int player) const {
        const int roster_size = scratch.roster_sizes[roster];
        const int position = players_[player].position;
        if (roster_size >= rounds_) {
            return false;
        }
        if (scratch.counts[roster][position] >= caps_[position]) {
            return false;
        }
        int missing = 0;
        for (int pos = 0; pos < kPositions; ++pos) {
            missing += std::max(0, required_[pos] - scratch.counts[roster][pos]);
        }
        int missing_after = missing -
            static_cast<int>(scratch.counts[roster][position] < required_[position]);
        int spare_flex = 0;
        for (const int pos : {kRB, kWR, kTE}) {
            spare_flex += std::max(0, scratch.counts[roster][pos] +
                static_cast<int>(position == pos) - required_[pos]);
        }
        missing_after += std::max(0, flex_slots_ - spare_flex);
        const int remaining_after = rounds_ - roster_size - 1;
        const int starter_count = std::accumulate(required_.begin(), required_.end(), flex_slots_);
        return missing_after <= remaining_after + std::max(0, starter_count - rounds_);
    }

    double pick_score(
        const Scratch& scratch,
        int roster,
        int player,
        int round,
        double rank,
        const std::array<double, kPositions>* bias,
        int current_pick
    ) const {
        const int position = players_[player].position;
        const auto& counts = scratch.counts[roster];
        double score = rank;
        if (counts[position] < required_[position]) {
            score -= 4.0;
        }
        if (position == kRB || position == kWR) {
            const int flex_target = required_[position] + (flex_slots_ + 1) / 2;
            if (counts[position] < flex_target) {
                score -= 2.0;
            }
            const int other = position == kRB ? kWR : kRB;
            const int imbalance = counts[other] - counts[position];
            score -= static_cast<double>(std::max(0, imbalance - 1) * 10);
        } else if (position == kTE && counts[kTE] >= std::max(1, required_[kTE])) {
            score += 15.0 * (counts[kTE] - std::max(1, required_[kTE]) + 1);
        } else if (position == kQB) {
            if (!has_projections_) {
                score += static_cast<double>(std::max(0, 12 - teams_) * 2);
                score -= std::max(0.0, pass_td_ - 4.0) * 3.0;
            }
            if (counts[kQB] >= std::max(1, required_[kQB])) {
                score += (round < 12 ? 30.0 : 18.0) +
                    40.0 * (counts[kQB] - std::max(1, required_[kQB]));
            }
        } else if ((position == kK || position == kDEF) && round < rounds_ - 2) {
            score += 85.0;
        }
        if ((position == kK || position == kDEF) && counts[position] >= required_[position]) {
            score += 100.0 * (counts[position] - required_[position] + 1);
        }
        if (bias != nullptr) {
            score += (*bias)[position];
        } else {
            const double next_availability = next_turn_availability(player, current_pick);
            const double urgency = (1.0 - next_availability) *
                players_[player].tier_gap_rank;
            score -= std::min(12.0, urgency);
        }
        return score;
    }

    int select_player(
        const Scratch& scratch,
        int roster,
        const std::vector<int>& order,
        const std::vector<double>& ranks,
        const std::array<double, kPositions>* bias
    ) const {
        int best_player = -1;
        double best_score = std::numeric_limits<double>::infinity();
        int legal_seen = 0;
        const int round = scratch_current_pick_ / teams_ + 1;
        for (const int player : order) {
            if (scratch.drafted[player] || !can_add(scratch, roster, player)) {
                continue;
            }
            const double score = pick_score(
                scratch, roster, player, round, ranks[player], bias,
                scratch_current_pick_
            );
            if (score < best_score) {
                best_score = score;
                best_player = player;
            }
            if (++legal_seen >= 40) {
                break;
            }
        }
        return best_player;
    }

    void draft_player(Scratch& scratch, int roster, int player) const {
        const int size = scratch.roster_sizes[roster];
        scratch.roster_players[roster * rounds_ + size] = player;
        scratch.roster_sizes[roster] = size + 1;
        ++scratch.counts[roster][players_[player].position];
        scratch.drafted[player] = 1;
    }

    void sample_board(Scratch& scratch, CounterRng& rng) const {
        for (int index = 0; index < static_cast<int>(players_.size()); ++index) {
            scratch.sampled_ranks[index] = std::clamp(
                players_[index].adp_mean + players_[index].adp_sigma * rng.normal(),
                1.0,
                500.0
            );
            scratch.order[index] = index;
        }
        std::sort(scratch.order.begin(), scratch.order.end(), [&](int left, int right) {
            if (scratch.sampled_ranks[left] == scratch.sampled_ranks[right]) {
                return left < right;
            }
            return scratch.sampled_ranks[left] < scratch.sampled_ranks[right];
        });
    }

    void sample_outcomes(Scratch& scratch, CounterRng& rng) const {
        for (int index = 0; index < static_cast<int>(players_.size()); ++index) {
            scratch.outcomes[index] = std::exp(
                players_[index].outcome_mu + players_[index].outcome_sigma * rng.normal()
            );
        }
    }

    double lineup_score(const Scratch& scratch, int roster) const {
        std::array<std::uint8_t, 64> used{};
        const int size = scratch.roster_sizes[roster];
        const int offset = roster * rounds_;
        double starter_score = 0.0;

        auto take_best = [&](auto eligible, int count) {
            for (int slot = 0; slot < count; ++slot) {
                int best_local = -1;
                double best_points = -1.0;
                for (int local = 0; local < size; ++local) {
                    const int player = scratch.roster_players[offset + local];
                    if (!used[local] && eligible(players_[player].position) &&
                        players_[player].projected_points > best_points) {
                        best_points = players_[player].projected_points;
                        best_local = local;
                    }
                }
                if (best_local >= 0) {
                    used[best_local] = 1;
                    starter_score += scratch.outcomes[scratch.roster_players[offset + best_local]];
                }
            }
        };

        for (int position = 0; position < kPositions; ++position) {
            take_best([&](int candidate_position) {
                return candidate_position == position;
            }, required_[position]);
        }
        take_best([](int position) {
            return position == kRB || position == kWR || position == kTE;
        }, flex_slots_);

        std::array<std::pair<double, double>, 64> bench{};
        int bench_size = 0;
        for (int local = 0; local < size; ++local) {
            if (used[local]) {
                continue;
            }
            const int player = scratch.roster_players[offset + local];
            const int position = players_[player].position;
            if (position == kQB || position == kRB || position == kWR || position == kTE) {
                bench[bench_size++] = {players_[player].projected_points, scratch.outcomes[player]};
            }
        }
        std::stable_sort(bench.begin(), bench.begin() + bench_size, [](const auto& left, const auto& right) {
            return left.first > right.first;
        });
        double bench_score = 0.0;
        for (int index = 0; index < std::min(4, bench_size); ++index) {
            bench_score += bench[index].second;
        }
        return (starter_score + 0.08 * bench_score) / 17.0;
    }

    RolloutSummary rollout(
        Scratch& scratch,
        std::uint64_t seed,
        int forced_player,
        std::vector<std::uint64_t>* availability
    ) const {
        scratch.reset();
        // Separate streams keep every stochastic input keyed by rollout and draw
        // type. Forced candidates therefore see the same board and outcomes.
        CounterRng board_rng(CounterRng::mix(seed ^ 0x243f6a8885a308d3ULL));
        CounterRng outcome_rng(CounterRng::mix(seed ^ 0x13198a2e03707344ULL));
        sample_board(scratch, board_rng);
        sample_outcomes(scratch, outcome_rng);
        int first_selection = -1;
        const int first_user_pick = next_user_pick(next_pick_);

        for (scratch_current_pick_ = next_pick_; scratch_current_pick_ < teams_ * rounds_;
             ++scratch_current_pick_) {
            const int roster = pick_owners_[scratch_current_pick_];
            if (scratch_current_pick_ == first_user_pick && availability != nullptr) {
                for (const int player : value_order_) {
                    if (!scratch.drafted[player]) {
                        ++(*availability)[player];
                    }
                }
            }

            int player = -1;
            if (roster == user_roster_) {
                if (scratch_current_pick_ == first_user_pick && forced_player >= 0) {
                    if (scratch.drafted[forced_player]) {
                        return RolloutSummary{.failed = true};
                    }
                    player = forced_player;
                } else {
                    player = select_player(
                        scratch, roster, value_order_, value_ranks_, nullptr
                    );
                }
                if (first_selection < 0) {
                    first_selection = player;
                }
            } else {
                player = select_player(
                    scratch, roster, scratch.order, scratch.sampled_ranks, &biases_[roster]
                );
            }
            if (player < 0) {
                return RolloutSummary{.failed = true};
            }
            draft_player(scratch, roster, player);
        }

        double user_score = 0.0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int roster = 0; roster < teams_; ++roster) {
            const double score = lineup_score(scratch, roster);
            if (roster == user_roster_) {
                user_score = score;
            }
            best_score = std::max(best_score, score);
        }
        return RolloutSummary{
            .first_selection = first_selection,
            .user_score = user_score,
            .top_roster = user_score >= best_score,
            .failed = false,
        };
    }

    int next_user_pick(int start) const {
        for (int pick = start; pick < teams_ * rounds_; ++pick) {
            if (pick_owners_[pick] == user_roster_) {
                return pick;
            }
        }
        return teams_ * rounds_;
    }

    double next_turn_availability(int player, int current_pick) const {
        const int following_pick = next_user_pick(current_pick + 1);
        if (following_pick >= teams_ * rounds_) {
            return 1.0;
        }
        if (following_pick == current_pick + 1) {
            return 1.0;
        }
        const auto& data = players_[player];
        const auto log_survival = [&](double pick_number) {
            const double z = (pick_number - 0.5 - data.adp_mean) / data.adp_sigma;
            if (z < 8.0) {
                return std::log(0.5 * std::erfc(z / std::sqrt(2.0)));
            }
            const double inv2 = 1.0 / (z * z);
            const double correction = 1.0 - inv2 + 3.0 * inv2 * inv2 -
                15.0 * inv2 * inv2 * inv2 + 105.0 * inv2 * inv2 * inv2 * inv2;
            return -0.5 * z * z - std::log(z) - 0.5 * std::log(2.0 * kPi) + std::log(correction);
        };
        return std::exp(std::min(0.0,
            log_survival(following_pick + 1.0) - log_survival(current_pick + 1.0)));
    }

    OnClockResult run_on_clock(
        std::uint64_t total,
        const std::vector<int>& candidates,
        std::uint64_t seed,
        int requested_threads
    ) const {
        OnClockResult result;
        result.candidate_indices = candidates;
        result.score_sums.assign(candidates.size(), 0.0);
        result.top_counts.assign(candidates.size(), 0);
        result.samples.assign(candidates.size(), 0);
        if (candidates.empty()) {
            return result;
        }
        if (total < candidates.size()) {
            throw std::invalid_argument(
                "total_rollouts must be at least the number of candidates"
            );
        }

        const std::uint64_t base = total / candidates.size();
        const std::uint64_t remainder = total % candidates.size();
        std::vector<std::uint64_t> starts(candidates.size() + 1, 0);
        for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
            starts[candidate + 1] = starts[candidate] + base +
                static_cast<std::uint64_t>(candidate < remainder);
        }
        const int thread_count = resolve_threads(requested_threads, total);
        struct Local {
            std::vector<double> scores;
            std::vector<std::uint64_t> tops;
            std::vector<std::uint64_t> samples;
            bool failed{};
        };
        std::vector<Local> locals(thread_count);
        for (auto& local : locals) {
            local.scores.assign(candidates.size(), 0.0);
            local.tops.assign(candidates.size(), 0);
            local.samples.assign(candidates.size(), 0);
        }
        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        for (int thread = 0; thread < thread_count; ++thread) {
            threads.emplace_back([&, thread] {
                Scratch scratch(static_cast<int>(players_.size()), teams_, rounds_);
                load_base(scratch);
                auto& local = locals[thread];
                for (std::uint64_t job = thread; job < total; job += thread_count) {
                    const auto upper = std::upper_bound(starts.begin(), starts.end(), job);
                    const std::size_t candidate = static_cast<std::size_t>(upper - starts.begin() - 1);
                    const std::uint64_t run = job - starts[candidate];
                    const auto summary = rollout(
                        scratch, rollout_seed(seed, run), candidates[candidate], nullptr
                    );
                    if (summary.failed) {
                        local.failed = true;
                        continue;
                    }
                    local.scores[candidate] += summary.user_score;
                    local.tops[candidate] += static_cast<std::uint64_t>(summary.top_roster);
                    ++local.samples[candidate];
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (const auto& local : locals) {
            if (local.failed) {
                throw std::runtime_error("native rollout could not select a legal player");
            }
            for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
                result.score_sums[candidate] += local.scores[candidate];
                result.top_counts[candidate] += local.tops[candidate];
                result.samples[candidate] += local.samples[candidate];
            }
        }
        result.total_rollouts = std::accumulate(
            result.samples.begin(), result.samples.end(), std::uint64_t{0}
        );
        return result;
    }

    BeforeTurnResult run_before_turn(
        std::uint64_t total,
        std::uint64_t seed,
        int requested_threads
    ) const {
        const int thread_count = resolve_threads(requested_threads, total);
        struct Local {
            explicit Local(std::size_t players)
                : selected(players), available(players), score_sums(players), tops(players) {}
            std::vector<std::uint64_t> selected;
            std::vector<std::uint64_t> available;
            std::vector<double> score_sums;
            std::vector<std::uint64_t> tops;
            bool failed{};
        };
        std::vector<Local> locals;
        locals.reserve(thread_count);
        for (int thread = 0; thread < thread_count; ++thread) {
            locals.emplace_back(players_.size());
        }
        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        for (int thread = 0; thread < thread_count; ++thread) {
            threads.emplace_back([&, thread] {
                Scratch scratch(static_cast<int>(players_.size()), teams_, rounds_);
                load_base(scratch);
                auto& local = locals[thread];
                for (std::uint64_t run = thread; run < total; run += thread_count) {
                    const auto summary = rollout(
                        scratch, rollout_seed(seed, run), -1, &local.available
                    );
                    if (summary.failed || summary.first_selection < 0) {
                        local.failed = true;
                        continue;
                    }
                    ++local.selected[summary.first_selection];
                    local.score_sums[summary.first_selection] += summary.user_score;
                    local.tops[summary.first_selection] +=
                        static_cast<std::uint64_t>(summary.top_roster);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        BeforeTurnResult result;
        result.selected_counts.assign(players_.size(), 0);
        result.available_counts.assign(players_.size(), 0);
        result.score_sums.assign(players_.size(), 0.0);
        result.top_counts.assign(players_.size(), 0);
        for (const auto& local : locals) {
            if (local.failed) {
                throw std::runtime_error("native rollout could not select a legal player");
            }
            for (std::size_t player = 0; player < players_.size(); ++player) {
                result.selected_counts[player] += local.selected[player];
                result.available_counts[player] += local.available[player];
                result.score_sums[player] += local.score_sums[player];
                result.top_counts[player] += local.tops[player];
            }
        }
        result.total_rollouts = total;
        return result;
    }

    int teams_{};
    int rounds_{};
    int flex_slots_{};
    int user_roster_{};
    int next_pick_{};
    bool has_projections_{};
    double pass_td_{};
    std::array<int, kPositions> required_{};
    std::array<int, kPositions> caps_{};
    std::vector<PlayerData> players_;
    std::vector<int> value_order_;
    std::vector<double> value_ranks_;
    std::vector<int> pick_owners_;
    std::vector<std::array<double, kPositions>> biases_;
    std::vector<int> base_roster_players_;
    std::vector<int> base_roster_sizes_;
    std::vector<std::array<int, kPositions>> base_counts_;
    std::vector<std::uint8_t> base_drafted_;
    // Each rollout is owned by one thread, so this avoids plumbing the current pick
    // through every hot helper without sharing state between engine instances.
    static thread_local int scratch_current_pick_;
};

thread_local int NativeDraftEngine::scratch_current_pick_ = 0;

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native C++23 Monte Carlo draft engine";
    py::class_<NativeDraftEngine>(module, "NativeDraftEngine")
        .def(py::init<const py::dict&>())
        .def("set_state", &NativeDraftEngine::set_state)
        .def("candidate_indices", &NativeDraftEngine::candidate_indices)
        .def("recommend_on_clock", &NativeDraftEngine::recommend_on_clock)
        .def("recommend_before_turn", &NativeDraftEngine::recommend_before_turn);
}
