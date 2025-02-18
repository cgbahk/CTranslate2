#include "ctranslate2/vocabulary_map.h"

namespace ctranslate2 {

  VocabularyMap::VocabularyMap(std::istream& map_file, const Vocabulary& vocabulary)
    : _vocabulary_size(vocabulary.size()) {
    std::string line;
    while (std::getline(map_file, line)) {
      std::string token;
      std::string key;
      std::vector<size_t> values;
      bool target = false;
      size_t ngram = 1;

      for (size_t i = 0; i < line.length(); ++i) {
        if (line[i] == '\t') {
          target = true;
          std::swap(key, token);
        } else if (line[i] == ' ') {
          if (target) {
            values.push_back(vocabulary.to_id(token));
            token.clear();
          } else {
            token += line[i];
            ++ngram;
          }
        } else
          token += line[i];
      }

      if (!token.empty())
        values.push_back(vocabulary.to_id(token));

      if (ngram > _map_rules.size())
        _map_rules.resize(ngram);

      _map_rules[ngram - 1][key] = values;
    }

    _fixed_candidates.insert(vocabulary.unk_id());
    _fixed_candidates.insert(vocabulary.bos_id());
    _fixed_candidates.insert(vocabulary.eos_id());

    // The field marked by the empty string are common tokens that are always candidates.
    auto it = _map_rules[0].find("");
    if (it != _map_rules[0].end())
      _fixed_candidates.insert(it->second.begin(), it->second.end());
  }

  std::vector<size_t>
  VocabularyMap::get_candidates(const std::vector<std::vector<std::string>>& batch_tokens) const {
    std::set<size_t> candidates = _fixed_candidates;
    std::string accu;
    for (const auto& tokens : batch_tokens) {
      for (size_t i = 0; i < tokens.size(); i++) {
        accu.clear();
        for (size_t h = 0; h < _map_rules.size() && i + h < tokens.size(); ++h) {
          if (h > 0)
            accu += ' ';
          accu += tokens[i + h];
          auto it = _map_rules[h].find(accu);
          if (it != _map_rules[h].end())
            candidates.insert(it->second.begin(), it->second.end());
        }
      }
    }

    return std::vector<size_t>(candidates.begin(), candidates.end());
  }

}
