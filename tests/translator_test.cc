#include <ctranslate2/buffered_translation_wrapper.h>
#include <ctranslate2/decoding.h>

#include <algorithm>
#include <unordered_set>

#include "test_utils.h"

static std::string
path_to_test_name(::testing::TestParamInfo<std::pair<std::string, DataType>> param_info) {
  std::string name = param_info.param.first;
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  return name;
}

static std::string beam_to_test_name(::testing::TestParamInfo<size_t> param_info) {
  if (param_info.param == 1)
    return "GreedySearch";
  else
    return "BeamSearch";
}

static void check_weights_dtype(const std::unordered_map<std::string, StorageView>& variables,
                                DataType expected_dtype) {
  for (const auto& variable : variables) {
    const auto& name = variable.first;
    const auto& value = variable.second;
    if (ends_with(name, "weight")) {
      EXPECT_EQ(value.dtype(), expected_dtype) << "Expected type " << dtype_name(expected_dtype)
                                               << " for weight " << name << ", got "
                                               << dtype_name(value.dtype()) << " instead";
    }
  }
}

static DataType dtype_with_fallback(DataType dtype, Device device) {
  const bool support_int8 = mayiuse_int8(device);
  const bool support_int16 = mayiuse_int16(device);
  if (dtype == DataType::INT16 && !support_int16)
    return support_int8 ? DataType::INT8 : DataType::FLOAT;
  if (dtype == DataType::INT8 && !support_int8)
    return support_int16 ? DataType::INT16 : DataType::FLOAT;
  return dtype;
}


// Test that we can load and translate with different versions of the same model.
class ModelVariantTest : public ::testing::TestWithParam<std::pair<std::string, DataType>> {
};

TEST_P(ModelVariantTest, Transliteration) {
  auto params = GetParam();
  const std::string model_path = get_data_dir() + "/models/" + params.first;
  const DataType model_dtype = params.second;
  const Device device = Device::CPU;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> expected = {"a", "t", "z", "m", "o", "n"};

  std::vector<std::pair<ComputeType, DataType>> type_params;
  type_params.emplace_back(ComputeType::DEFAULT, dtype_with_fallback(model_dtype, device));
  type_params.emplace_back(ComputeType::FLOAT, DataType::FLOAT);
  if (mayiuse_int16(device))
    type_params.emplace_back(ComputeType::INT16, DataType::INT16);
  if (mayiuse_int8(device)) {
    type_params.emplace_back(ComputeType::INT8, DataType::INT8);
    type_params.emplace_back(ComputeType::AUTO, DataType::INT8);
  } else if (mayiuse_int16(device)) {
    type_params.emplace_back(ComputeType::AUTO, DataType::INT16);
  } else {
    type_params.emplace_back(ComputeType::AUTO, DataType::FLOAT);
  }

  for (const auto& types : type_params) {
    const ComputeType compute_type = types.first;
    const DataType expected_type = types.second;
    const auto model = models::Model::load(model_path, device, 0, compute_type);
    check_weights_dtype(model->get_variables(), expected_type);
    Translator translator(model);
    auto result = translator.translate(input);
    EXPECT_EQ(result.output(), expected);
  }
}

INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  ModelVariantTest,
  ::testing::Values(
    std::make_pair("v1/aren-transliteration", DataType::FLOAT),
    std::make_pair("v1/aren-transliteration-i16", DataType::INT16),
    std::make_pair("v2/aren-transliteration", DataType::FLOAT),
    std::make_pair("v2/aren-transliteration-i16", DataType::INT16),
    std::make_pair("v2/aren-transliteration-i8", DataType::INT8)
    ),
  path_to_test_name);


class SearchVariantTest : public ::testing::TestWithParam<size_t> {
};

static Translator default_translator(Device device = Device::CPU) {
  return Translator(default_model_dir(), device);
}

TEST_P(SearchVariantTest, SetMaxDecodingLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.max_decoding_length = 3;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.output().size(), options.max_decoding_length);
}

TEST_P(SearchVariantTest, SetMinDecodingLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.min_decoding_length = 8;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.output().size(), options.min_decoding_length);
}

TEST_P(SearchVariantTest, SetMaxInputLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.max_input_length = 3;
  options.return_attention = true;

  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);

  EXPECT_EQ(result.hypotheses[0].size(), options.max_input_length);

  // Check that attention vectors have the size of the original input.
  ASSERT_TRUE(result.has_attention());
  for (size_t i = 0; i < result.attention[0].size(); ++i) {
    ASSERT_EQ(result.attention[0][i].size(), input.size());
    for (size_t t = options.max_input_length; t < input.size(); ++t) {
      EXPECT_EQ(result.attention[0][i][t], 0);
    }
  }
}

TEST_P(SearchVariantTest, ReturnAllHypotheses) {
  auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.num_hypotheses(), beam_size);
}

TEST_P(SearchVariantTest, ReturnAttention) {
  auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.return_attention = true;
  const std::vector<std::vector<std::string>> inputs = {
    {"آ", "ز", "ا"},
    {"آ", "ت", "ز", "م", "و", "ن"}
  };
  const std::vector<std::pair<size_t, size_t>> expected_shapes = {
    {4, 3},
    {6, 6}
  };  // (target_length, source_length)
  const auto results = translator.translate_batch(inputs, options);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const TranslationResult& result = results[i];
    const auto& expected_shape = expected_shapes[i];
    ASSERT_TRUE(result.has_attention());
    const auto& attention = result.attention;
    EXPECT_EQ(attention.size(), beam_size);
    EXPECT_EQ(attention[0].size(), expected_shape.first);
    for (const auto& vector : attention[0]) {
      EXPECT_EQ(vector.size(), expected_shape.second);
    }
  }
}

TEST_P(SearchVariantTest, ReturnAttentionWithPrefix) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.return_attention = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"<unk>", "t"};
  std::vector<std::string> expected = {"<unk>", "t", "z", "m", "o", "n" };
  auto result = translator.translate_with_prefix(input, prefix, options);
  EXPECT_EQ(result.output(), expected);
  EXPECT_TRUE(result.has_attention());
  for (const auto& vector : result.attention[0]) {
    EXPECT_EQ(vector.size(), input.size());
  }
}

TEST_P(SearchVariantTest, TranslateWithPrefix) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.return_attention = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"a", "t", "s"};
  std::vector<std::string> expected = {"a", "t", "s", "u", "m", "o", "n"};
  auto result = translator.translate_with_prefix(input, prefix, options);
  EXPECT_EQ(result.num_hypotheses(), beam_size);
  EXPECT_EQ(result.output(), expected);
  ASSERT_TRUE(result.has_attention());
  const auto& attention = result.attention;
  EXPECT_EQ(attention.size(), options.beam_size);
  EXPECT_EQ(attention[0].size(), 7);
  EXPECT_EQ(attention[0][0].size(), 6);
}

TEST_P(SearchVariantTest, TranslateBatch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.return_scores = true;
  options.beam_size = GetParam();
  std::vector<std::vector<std::string>> inputs = {
    {"آ", "ز", "ا"},
    {"آ", "ت", "ز", "م", "و", "ن"}};
  std::vector<std::vector<std::string>> expected = {
    {"a", "z", "z", "a"},
    {"a", "t", "z", "m", "o", "n"}};
  auto result = translator.translate_batch(inputs, options);
  EXPECT_TRUE(result[0].has_scores());
  EXPECT_TRUE(result[1].has_scores());
  EXPECT_EQ(result[0].output(), expected[0]);
  EXPECT_EQ(result[1].output(), expected[1]);
}

TEST_P(SearchVariantTest, ReplaceUnknowns) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.replace_unknowns = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"<unk>", "t"};
  std::vector<std::string> expected = {"ت", "t", "z", "m", "o", "n" };
  auto result = translator.translate_with_prefix(input, prefix, options);
  EXPECT_EQ(result.output(), expected);
}

TEST_P(SearchVariantTest, RepetitionPenalty) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.repetition_penalty = 100;  // Force the decoding to produce unique symbols.
  const auto result = translator.translate({"ن", "ن", "ن", "ن", "ن"}, options);
  const auto& tokens = result.output();
  const std::unordered_set<std::string> unique_tokens(tokens.begin(), tokens.end());
  EXPECT_EQ(unique_tokens.size(), tokens.size());
}

static void check_normalized_score(const std::vector<std::string>& input,
                                   TranslationOptions options,
                                   bool output_has_eos = true) {
  Translator translator = default_translator();
  options.return_scores = true;
  const auto score = translator.translate(input, options).scores[0];

  options.normalize_scores = true;
  const auto normalized_result = translator.translate(input, options);
  const auto normalized_score = normalized_result.scores[0];
  auto normalized_length = normalized_result.hypotheses[0].size();
  if (output_has_eos)
    normalized_length += 1;

  EXPECT_NEAR(normalized_score, score / normalized_length, 1e-6);
}

TEST_P(SearchVariantTest, NormalizeScores) {
  TranslationOptions options;
  options.beam_size = GetParam();
  check_normalized_score({"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}, options);
}

TEST_P(SearchVariantTest, NormalizeScoresNoEos) {
  TranslationOptions options;
  options.beam_size = GetParam();
  options.max_decoding_length = 6;
  check_normalized_score({"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}, options, false);
}

TEST_P(SearchVariantTest, NormalizeScoresAlternatives) {
  TranslationOptions options;
  options.beam_size = GetParam();
  options.return_alternatives = true;
  check_normalized_score({"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}, options);
}

TEST_P(SearchVariantTest, NormalizeScoresAlternativesNoEos) {
  TranslationOptions options;
  options.beam_size = GetParam();
  options.return_alternatives = true;
  options.max_decoding_length = 6;
  check_normalized_score({"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}, options, false);
}


INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  SearchVariantTest,
  ::testing::Values(1, 4),
  beam_to_test_name);

TEST(TranslatorTest, TranslateEmptyBatch) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs;
  auto results = translator.translate_batch(inputs);
  EXPECT_TRUE(results.empty());
}

static void check_empty_result(const TranslationResult& result,
                               size_t num_hypotheses = 1,
                               bool with_attention = false,
                               bool with_score = false) {
  EXPECT_TRUE(result.output().empty());
  EXPECT_EQ(result.num_hypotheses(), num_hypotheses);
  EXPECT_EQ(result.hypotheses.size(), num_hypotheses);
  EXPECT_EQ(result.has_scores(), with_score);
  if (with_score) {
    EXPECT_EQ(result.scores.size(), num_hypotheses);
    EXPECT_EQ(result.score(), 0);
    for (const auto score : result.scores) {
      EXPECT_EQ(score, 0);
    }
  }
  EXPECT_EQ(result.has_attention(), with_attention);
  if (with_attention) {
    const auto& attention = result.attention;
    EXPECT_EQ(attention.size(), num_hypotheses);
    EXPECT_TRUE(attention[0].empty());
  }
}

TEST(TranslatorTest, TranslateBatchWithEmptySource) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs = {
    {}, {"آ", "ز", "ا"}, {}, {"آ", "ت", "ز", "م", "و", "ن"}, {}};
  auto results = translator.translate_batch(inputs);
  EXPECT_EQ(results.size(), 5);
  check_empty_result(results[0]);
  EXPECT_EQ(results[1].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
  check_empty_result(results[2]);
  EXPECT_EQ(results[3].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  check_empty_result(results[4]);
}

TEST(TranslatorTest, TranslateBatchWithOnlyEmptySource) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs{{}, {}};
  auto results = translator.translate_batch(inputs);
  EXPECT_EQ(results.size(), 2);
  check_empty_result(results[0]);
  check_empty_result(results[1]);
}

TEST(TranslatorTest, TranslateEmptySourceWithoutScore) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.return_scores = false;
  EXPECT_FALSE(translator.translate(std::vector<std::string>{}, options).has_scores());
}

TEST(TranslatorTest, TranslateBatchWithHardPrefixAndEmpty) {
  Translator translator = default_translator();
  const TranslationOptions options;
  const std::vector<std::vector<std::string>> input = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {},
    {"آ" ,"ز" ,"ا"}};
  const std::vector<std::vector<std::string>> prefix = {
    {"a", "t", "s"},
    {},
    {"a", "t", "z", "o"},
    {},
    {}};
  const auto result = translator.translate_batch_with_prefix(input, prefix, options);
  EXPECT_EQ(result[0].output(), (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));
  EXPECT_EQ(result[1].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[2].output(), (std::vector<std::string>{"a", "t", "z", "o", "m", "o", "n"}));
  EXPECT_TRUE(result[3].output().empty());
  EXPECT_EQ(result[4].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
}

TEST(TranslatorTest, TranslateBatchWithStronglyBiasedPrefix) {
  // This test should produce the same results as TranslateBatchWithHardPrefixAndEmpty
  // because prefix_bias_beta is set to 0.99, which is almost equivalent to using a hard prefix.
  Translator translator = default_translator();
  TranslationOptions options;
  options.prefix_bias_beta = 0.99;
  options.beam_size = 2;
  const std::vector<std::vector<std::string>> input = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ز" ,"ا"}};
  const std::vector<std::vector<std::string>> prefix = {
    {"a", "t", "s"},
    {},
    {"a", "t", "z", "o"},
    {}};
  const auto result = translator.translate_batch_with_prefix(input, prefix, options);
  EXPECT_EQ(result[0].output(), (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));
  EXPECT_EQ(result[1].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[2].output(), (std::vector<std::string>{"a", "t", "z", "o", "m", "o", "n"}));
  EXPECT_EQ(result[3].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
}

TEST(TranslatorTest, TranslateBatchWithWeaklyBiasedPrefix) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.prefix_bias_beta = 0.01;
  options.beam_size = 2;
  const std::vector<std::vector<std::string>> input = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ز" ,"ا"}
  };
  const std::vector<std::vector<std::string>> prefix = {
    {"a", "t", "s", "s", "s"},   // Test divergence at divergence first 's'
    {},
    {"a", "t", "z", "o"},
    {}
  };
  const auto result = translator.translate_batch_with_prefix(input, prefix, options);
  EXPECT_EQ(result[0].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[1].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[2].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[3].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
}

class BiasedDecodingDeviceFPTest : public ::testing::TestWithParam<std::pair<Device, DataType>> {
};

TEST_P(BiasedDecodingDeviceFPTest, OneBatchOneBeam) {
    const Device device = GetParam().first;
    const DataType dtype = GetParam().second;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size},
                       std::vector<float>{4, 6});
    StorageView softmax;
    ops::SoftMax()(logits, softmax);
    std::vector<float> expected_log_probs_vec = {
      std::log((1-prefix_bias_beta) * softmax.at<float>(0) + prefix_bias_beta),
      std::log((1-prefix_bias_beta) * softmax.at<float>(1)),
    };
    StorageView expected_log_probs(logits.shape(), expected_log_probs_vec, device);

    StorageView log_probs(device, dtype);
    const size_t step = 0;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{false}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0}};
    ctranslate2::BiasedDecoder biased_decoder;
    biased_decoder.decode(prefix_bias_beta,
                          batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          prefix_ids,
                          logits.to(device).to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, TwoBatchesTwoBeams) {
    const Device device = GetParam().first;
    const DataType dtype = GetParam().second;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 2;
    const dim_t beam_size = 2;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{
        4, 6,  // batch1 beam1
        7, 3,  // batch1 beam2
        1, 9,  // batch2 beam1
        8, 2   // batch2 beam2
    });
    StorageView softmax;
    ops::SoftMax()(logits, softmax);
    const std::vector<std::vector<size_t>> prefix_ids = {
      {0},  // bias batch 1 towards token0
      {1}}; // bias batch 2 towards token1
    std::vector<float> expected_log_probs_vec = {
      //batch1 beam1
      std::log((1-prefix_bias_beta) * softmax.at<float>(0) + prefix_bias_beta),
      std::log((1-prefix_bias_beta) * softmax.at<float>(1)),
      //batch1 beam2
      std::log((1-prefix_bias_beta) * softmax.at<float>(2) + prefix_bias_beta),
      std::log((1-prefix_bias_beta) * softmax.at<float>(3)),
      //batch2 beam1
      std::log((1-prefix_bias_beta) * softmax.at<float>(4)),
      std::log((1-prefix_bias_beta) * softmax.at<float>(5) + prefix_bias_beta),
      //batch2 beam2
      std::log((1-prefix_bias_beta) * softmax.at<float>(6)),
      std::log((1-prefix_bias_beta) * softmax.at<float>(7) + prefix_bias_beta),
    };
    StorageView expected_log_probs(logits.shape(), expected_log_probs_vec, device);

    StorageView log_probs(dtype, device);
    const size_t step = 0;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{false, false}, {false, false}};
    const std::vector<dim_t> batch_offset = {0, 1};
    BiasedDecoder biased_decoder;
    biased_decoder.decode(prefix_bias_beta,
                          batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          prefix_ids,
                          logits.to(device).to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, BeamDiverged) {
    const Device device = GetParam().first;
    const DataType dtype = GetParam().second;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{4, 6}, device);
    StorageView expected_log_probs(device);
    ops::LogSoftMax()(logits, expected_log_probs);

    StorageView log_probs(dtype, device);
    const size_t step = 0;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{true}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0}};
    BiasedDecoder biased_decoder;
    biased_decoder.decode(prefix_bias_beta,
                          batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          prefix_ids,
                          logits.to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, TimeStepPastPrefix) {
    const Device device = GetParam().first;
    const DataType dtype = GetParam().second;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{4, 6}, device);
    StorageView expected_log_probs(device);
    ops::LogSoftMax()(logits, expected_log_probs);

    StorageView log_probs(dtype, device);
    const size_t step = 1;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{false}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0}};
    BiasedDecoder biased_decoder;
    biased_decoder.decode(prefix_bias_beta,
                          batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          prefix_ids,
                          logits.to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, NonZeroTimestepBias) {
    const Device device = GetParam().first;
    const DataType dtype = GetParam().second;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{4, 6});
    StorageView softmax;
    ops::SoftMax()(logits, softmax);
    std::vector<float> expected_log_probs_vec = {
      std::log((1-prefix_bias_beta) * softmax.at<float>(0)),
      std::log((1-prefix_bias_beta) * softmax.at<float>(1) + prefix_bias_beta),
    };
    StorageView expected_log_probs(logits.shape(), expected_log_probs_vec, device);

    StorageView log_probs(dtype, device);
    const size_t step = 1;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{false}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0, 1, 0}};
    BiasedDecoder biased_decoder;
    biased_decoder.decode(prefix_bias_beta,
                          batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          prefix_ids,
                          logits.to(device).to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, NonZeroTimestepDiverge) {
    const Device device = GetParam().first;
    const DataType dtype = GetParam().second;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{4, 6}, device);
    StorageView expected_log_probs(device);
    ops::LogSoftMax()(logits, expected_log_probs);

    StorageView log_probs(dtype, device);
    const size_t step = 1;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{true}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0, 1, 0}};
    BiasedDecoder biased_decoder;
    biased_decoder.decode(prefix_bias_beta,
                          batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          prefix_ids,
                          logits.to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float(), expected_log_probs, 0.01);
}

static std::string fp_test_name(::testing::TestParamInfo<std::pair<Device, DataType>> param_info) {
  return dtype_name(param_info.param.second);
}
INSTANTIATE_TEST_CASE_P(CPU, BiasedDecodingDeviceFPTest,
                        ::testing::Values(std::make_pair(Device::CPU, DataType::FLOAT)),
                        fp_test_name);
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_CASE_P(CUDA, BiasedDecodingDeviceFPTest,
                        ::testing::Values(std::make_pair(Device::CUDA, DataType::FLOAT),
                                          std::make_pair(Device::CUDA, DataType::FLOAT16)),
                        fp_test_name);
#endif

TEST(TranslatorTest, TranslatePrefixWithLargeBeam) {
  // Related to issue https://github.com/OpenNMT/CTranslate2/issues/277
  // This is an example where </s> appears in the topk of the first unconstrained decoding
  // step and produces an incorrect hypothesis that dominates others.
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = 5;
  const std::vector<std::string> input = {"أ" ,"و" ,"ل" ,"ي" ,"س" ,"س"};
  const std::vector<std::string> prefix = {"u", "l", "i", "s", "e"};
  const auto result = translator.translate_with_prefix(input, prefix, options);
  EXPECT_EQ(result.output(), (std::vector<std::string>{"u", "l", "i", "s", "e", "s"}));
}

TEST(TranslatorTest, AlternativesFromPrefix) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  options.return_attention = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> prefix = {"a", "t"};
  const TranslationResult result = translator.translate_with_prefix(input, prefix, options);
  ASSERT_EQ(result.num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(result.hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result.hypotheses[1], (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));

  // Tokens at the first unconstrained decoding position should be unique.
  std::vector<std::string> tokens_at_position;
  tokens_at_position.reserve(options.num_hypotheses);
  for (const std::vector<std::string>& hypothesis : result.hypotheses)
    tokens_at_position.emplace_back(hypothesis[prefix.size()]);
  EXPECT_EQ(std::unique(tokens_at_position.begin(), tokens_at_position.end()),
            tokens_at_position.end());

  EXPECT_TRUE(result.has_attention());
  EXPECT_EQ(result.attention[0].size(), 6);
}

TEST(TranslatorTest, AlternativesFromPrefixBatch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  const std::vector<std::vector<std::string>> input = {
    {"آ", "ز", "ا"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}
  };
  const std::vector<std::vector<std::string>> prefix = {{"a"}, {"a", "t"}};
  const auto results = translator.translate_batch_with_prefix(input, prefix, options);
  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(results[0].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[0].hypotheses[0], (std::vector<std::string>{"a", "z", "z", "a"}));
  EXPECT_EQ(results[0].hypotheses[1], (std::vector<std::string>{"a", "s", "z", "a"}));
  ASSERT_EQ(results[1].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[1].hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(results[1].hypotheses[1], (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));
}

TEST(TranslatorTest, AlternativesFromScratch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const TranslationResult result = translator.translate(input, options);
  ASSERT_EQ(result.num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(result.hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
}

TEST(TranslatorTest, AlternativesFromScratchBatch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  const std::vector<std::vector<std::string>> inputs = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ", "ز", "ا"}
  };
  const std::vector<TranslationResult> results = translator.translate_batch(inputs, options);
  ASSERT_EQ(results.size(), inputs.size());
  ASSERT_EQ(results[0].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[0].hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(results[0].hypotheses[1], (std::vector<std::string>{"e", "t", "z", "m", "o", "n"}));
  ASSERT_EQ(results[1].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[1].hypotheses[0], (std::vector<std::string>{"a", "z", "z", "a"}));
  EXPECT_EQ(results[1].hypotheses[1], (std::vector<std::string>{"e", "z", "a"}));
}

TEST(TranslatorTest, AlternativesFromFullTarget) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 4;
  options.return_alternatives = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> target = {"a", "t", "z", "m", "o", "n"};
  const TranslationResult result = translator.translate_with_prefix(input, target, options);
  EXPECT_EQ(result.hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n", "e"}));
}

TEST(TranslatorTest, DetachModel) {
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  Translator translator = default_translator();
  translator.detach_model();
  EXPECT_THROW(translator.translate(input), std::runtime_error);
  Translator clone(translator);
  EXPECT_THROW(clone.translate(input), std::runtime_error);
  translator.set_model(models::Model::load(default_model_dir()));
  translator.translate(input);
}

TEST(TranslatorTest, InvalidNumHypotheses) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 0;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  EXPECT_THROW(translator.translate(input, options), std::invalid_argument);
}

TEST(TranslatorTest, IgnoreScore) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = 1;
  options.return_scores = false;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const TranslationResult result = translator.translate(input, options);
  EXPECT_FALSE(result.has_scores());
  EXPECT_EQ(result.output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
}

TEST(TranslatorTest, SameBeamAndGreedyScore) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.return_scores = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  options.beam_size = 1;
  const auto greedy_score = translator.translate(input, options).score();
  options.beam_size = 2;
  const auto beam_score = translator.translate(input, options).score();
  EXPECT_NEAR(greedy_score, beam_score, 1e-5);
}

TEST(BufferedTranslationWrapperTest, Basic) {
  auto translator_pool = std::make_shared<TranslatorPool>(/*num_translators=*/1,
                                                          /*num_threads_per_translator=*/2,
                                                          default_model_dir());
  BufferedTranslationWrapper wrapper(translator_pool,
                                     /*max_batch_size=*/32,
                                     /*batch_timeout_in_micros=*/5000);

  auto future1 = wrapper.translate_async({"آ", "ز", "ا"});
  auto future2 = wrapper.translate_async({"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"});

  EXPECT_EQ(future1.get().hypotheses[0],
            (std::vector<std::string>{"a", "z", "z", "a"}));
  EXPECT_EQ(future2.get().hypotheses[0],
            (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
}

TEST(TranslatorTest, Scoring) {
  const std::vector<std::vector<std::string>> source = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ش" ,"ي" ,"س" ,"و" ,"ن"},
    {"آ" ,"ر" ,"ب" ,"ا" ,"ك" ,"ه"},
    {},
    {"آ" ,"ر" ,"ث" ,"ر"},
  };
  const std::vector<std::vector<std::string>> target = {
    {"a", "t", "z", "m", "o", "n"},
    {"a", "c", "h", "i", "s", "o", "n"},
    {"a", "r", "b", "a", "k", "e"},
    {},
    {"a", "r", "t", "h", "e", "r"},
  };
  const std::vector<std::vector<float>> expected_scores = {
    {-0.106023, -0.065410, -0.056002, -0.447953, -0.230714, -0.092184, -0.063463},
    {-0.072660, -0.300309, -0.181187, -0.395671, -0.025631, -0.123466, -0.002034, -0.012639},
    {-0.103136, -0.089504, -0.063889, -0.007327, -0.452072, -0.060154, -0.016636},
    {0},
    {-0.076704, -0.036037, -0.029253, -0.030273, -0.149276, -0.002440, -0.003742},
  };
  constexpr float abs_diff = 1e-5;

  Translator translator = default_translator();
  const auto scores = translator.score_batch(source, target);

  ASSERT_EQ(scores.size(), expected_scores.size());
  for (size_t i = 0; i < scores.size(); ++i)
    expect_vector_eq(scores[i].tokens_score, expected_scores[i], abs_diff);
}

TEST(TranslatorTest, ScoringMaxInputLength) {
  const std::vector<std::string> source = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> target = {"a", "t", "z", "m", "o", "n"};

  ScoringOptions options;
  options.max_input_length = 4;
  Translator translator = default_translator();
  const auto result = translator.score_batch({source}, {target}, options)[0];

  EXPECT_EQ(result.tokens, (std::vector<std::string>{"a", "t", "z", "</s>"}));
  EXPECT_EQ(result.tokens_score.size(), options.max_input_length);
}
