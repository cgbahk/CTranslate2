#include "ctranslate2/translator_pool.h"

#include <spdlog/spdlog.h>

namespace ctranslate2 {

  static thread_local Translator* local_translator = nullptr;

  Translator* TranslatorPool::get_translator() {
    return local_translator;
  }

  class TranslatorWorker : public ReplicaWorker {
  public:
    TranslatorWorker(const std::shared_ptr<const models::Model>& model, size_t num_threads)
      : ReplicaWorker(model->device(), model->device_index(), num_threads)
      , _translator(model)
    {
    }

    Translator& translator() {
      return _translator;
    }

  protected:
    void initialize() override {
      ReplicaWorker::initialize();

      local_translator = &_translator;
    }

    void finalize() override {
      // The CUDA context is destroyed when the thread exits, so we clear the translation
      // resources now when the CUDA context is still active.
      _translator.detach_model();

      local_translator = nullptr;
    }

  private:
    Translator _translator;
  };


  TranslatorPool::TranslatorPool(size_t num_translators,
                                 size_t num_threads_per_translator,
                                 const std::string& model_dir,
                                 const Device device,
                                 const int device_index,
                                 const ComputeType compute_type,
                                 const long max_queued_batches)
    : ReplicaPool(create_workers<TranslatorWorker>(num_translators,
                                                   num_threads_per_translator,
                                                   model_dir,
                                                   device,
                                                   {device_index},
                                                   compute_type),
                  max_queued_batches)
  {
  }

  TranslatorPool::TranslatorPool(size_t num_translators,
                                 size_t num_threads_per_translator,
                                 models::ModelReader& model_reader,
                                 const Device device,
                                 const int device_index,
                                 const ComputeType compute_type,
                                 const long max_queued_batches)
    : ReplicaPool(create_workers<TranslatorWorker>(num_translators,
                                                   num_threads_per_translator,
                                                   model_reader,
                                                   device,
                                                   {device_index},
                                                   compute_type),
                  max_queued_batches)
  {
  }

  TranslatorPool::TranslatorPool(size_t num_translators_per_device,
                                 size_t num_threads_per_translator,
                                 const std::string& model_dir,
                                 const Device device,
                                 const std::vector<int>& device_indices,
                                 const ComputeType compute_type,
                                 const long max_queued_batches)
    : ReplicaPool(create_workers<TranslatorWorker>(num_translators_per_device,
                                                   num_threads_per_translator,
                                                   model_dir,
                                                   device,
                                                   device_indices,
                                                   compute_type),
                  max_queued_batches)
  {
  }

  TranslatorPool::TranslatorPool(size_t num_translators_per_device,
                                 size_t num_threads_per_translator,
                                 models::ModelReader& model_reader,
                                 const Device device,
                                 const std::vector<int>& device_indices,
                                 const ComputeType compute_type,
                                 const long max_queued_batches)
    : ReplicaPool(create_workers<TranslatorWorker>(num_translators_per_device,
                                                   num_threads_per_translator,
                                                   model_reader,
                                                   device,
                                                   device_indices,
                                                   compute_type),
                  max_queued_batches)
  {
  }

  std::vector<std::future<TranslationResult>>
  TranslatorPool::translate_batch_async(const std::vector<std::vector<std::string>>& source,
                                        const TranslationOptions& options,
                                        const size_t max_batch_size,
                                        const BatchType batch_type) {
    return translate_batch_async(source, {}, options, max_batch_size, batch_type);
  }

  std::vector<std::future<TranslationResult>>
  TranslatorPool::translate_batch_async(const std::vector<std::vector<std::string>>& source,
                                        const std::vector<std::vector<std::string>>& target_prefix,
                                        const TranslationOptions& options,
                                        const size_t max_batch_size,
                                        const BatchType batch_type) {
    return post_examples(load_examples({source, target_prefix}),
                         max_batch_size,
                         batch_type,
                         TranslationJobCreator(options));
  }

  std::vector<std::future<ScoringResult>>
  TranslatorPool::score_batch_async(const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>& target,
                                    const ScoringOptions& options,
                                    const size_t max_batch_size,
                                    const BatchType batch_type) {
    return post_examples(load_examples({source, target}),
                         max_batch_size,
                         batch_type,
                         ScoringJobCreator(options));
  }

  std::vector<TranslationResult>
  TranslatorPool::translate_batch(const std::vector<std::vector<std::string>>& source,
                                  const TranslationOptions& options,
                                  const size_t max_batch_size,
                                  const BatchType batch_type) {
    return translate_batch(source, {}, options, max_batch_size, batch_type);
  }

  template <typename T>
  std::vector<T> get_results_from_futures(std::vector<std::future<T>> futures) {
    std::vector<T> results;
    results.reserve(futures.size());
    for (auto& future : futures)
      results.emplace_back(future.get());
    return results;
  }

  std::vector<TranslationResult>
  TranslatorPool::translate_batch(const std::vector<std::vector<std::string>>& source,
                                  const std::vector<std::vector<std::string>>& target_prefix,
                                  const TranslationOptions& options,
                                  const size_t max_batch_size,
                                  const BatchType batch_type) {
    return get_results_from_futures(translate_batch_async(source,
                                                          target_prefix,
                                                          options,
                                                          max_batch_size,
                                                          batch_type));
  }

  std::vector<ScoringResult>
  TranslatorPool::score_batch(const std::vector<std::vector<std::string>>& source,
                              const std::vector<std::vector<std::string>>& target,
                              const ScoringOptions& options,
                              const size_t max_batch_size,
                              const BatchType batch_type) {
    return get_results_from_futures(score_batch_async(source, target, options, max_batch_size, batch_type));
  }

  TranslationStats TranslatorPool::consume_text_file(const std::string& source_file,
                                                     const std::string& output_file,
                                                     const TranslationOptions& options,
                                                     size_t max_batch_size,
                                                     size_t read_batch_size,
                                                     BatchType batch_type,
                                                     bool with_scores,
                                                     const std::string* target_file) {
    auto source = open_file<std::ifstream>(source_file);
    auto output = open_file<std::ofstream>(output_file);
    auto target = (target_file
                   ? std::make_unique<std::ifstream>(open_file<std::ifstream>(*target_file))
                   : nullptr);

    return consume_text_file(source,
                             output,
                             options,
                             max_batch_size,
                             read_batch_size,
                             batch_type,
                             with_scores,
                             target.get());
  }

  TranslationStats TranslatorPool::consume_text_file(std::istream& source,
                                                     std::ostream& output,
                                                     const TranslationOptions& options,
                                                     size_t max_batch_size,
                                                     size_t read_batch_size,
                                                     BatchType batch_type,
                                                     bool with_scores,
                                                     std::istream* target) {
    return consume_raw_text_file(source,
                                 target,
                                 output,
                                 split_tokens,
                                 split_tokens,
                                 join_tokens,
                                 options,
                                 max_batch_size,
                                 read_batch_size,
                                 batch_type,
                                 with_scores);
  }

  TranslationStats TranslatorPool::score_text_file(const std::string& source_file,
                                                   const std::string& target_file,
                                                   const std::string& output_file,
                                                   const ScoringOptions& options,
                                                   size_t max_batch_size,
                                                   size_t read_batch_size,
                                                   BatchType batch_type,
                                                   bool with_tokens_score) {
    auto source = open_file<std::ifstream>(source_file);
    auto target = open_file<std::ifstream>(target_file);
    auto output = open_file<std::ofstream>(output_file);
    return score_text_file(source,
                           target,
                           output,
                           options,
                           max_batch_size,
                           read_batch_size,
                           batch_type,
                           with_tokens_score);
  }

  TranslationStats TranslatorPool::score_text_file(std::istream& source,
                                                   std::istream& target,
                                                   std::ostream& output,
                                                   const ScoringOptions& options,
                                                   size_t max_batch_size,
                                                   size_t read_batch_size,
                                                   BatchType batch_type,
                                                   bool with_tokens_score) {
    return score_raw_text_file(source,
                               target,
                               output,
                               split_tokens,
                               split_tokens,
                               join_tokens,
                               options,
                               max_batch_size,
                               read_batch_size,
                               batch_type,
                               with_tokens_score);
  }

  size_t TranslatorPool::num_translators() const {
    return _thread_pool->num_threads();
  }

  std::vector<std::shared_ptr<const models::Model>> TranslatorPool::detach_models() {
    std::vector<std::shared_ptr<const models::Model>> models;
    models.reserve(num_translators());
    for (size_t i = 0; i < num_translators(); ++i) {
      auto& translator = static_cast<TranslatorWorker&>(_thread_pool->get_worker(i)).translator();
      models.emplace_back(translator.detach_model());
    }
    return models;
  }

  void TranslatorPool::set_models(const std::vector<std::shared_ptr<const models::Model>>& models) {
    if (models.size() != num_translators())
      throw std::invalid_argument("The number of models does not match the number "
                                  "of parallel translators");

    for (size_t i = 0; i < num_translators(); ++i) {
      auto& translator = static_cast<TranslatorWorker&>(_thread_pool->get_worker(i)).translator();
      translator.set_model(models[i]);
    }
  }

  void TranslatorPool::clear_cache() const {
    for (size_t i = 0; i < num_translators(); ++i) {
      auto* allocator = static_cast<TranslatorWorker&>(_thread_pool->get_worker(i)).allocator();
      if (allocator)
        allocator->clear_cache();
    }
  }


  class TranslationJob : public BatchJob<TranslationResult> {
  public:
    TranslationJob(TranslationOptions options, Batch batch)
      : BatchJob(std::move(batch))
      , _options(std::move(options))
    {
    }

  protected:
    std::vector<TranslationResult> get_results(const Batch& batch) const override {
      spdlog::debug("Running batch translation on {} examples", batch.num_examples());
      auto results = TranslatorPool::get_translator()->translate_batch_with_prefix(
        batch.get_stream(0),
        batch.get_stream(1),
        _options);
      spdlog::debug("Finished batch translation");
      return results;
    }

  private:
    const TranslationOptions _options;
  };

  TranslatorPool::TranslationJobCreator::TranslationJobCreator(TranslationOptions options)
    : _options(std::move(options))
  {
  }

  std::unique_ptr<BatchJob<TranslationResult>>
  TranslatorPool::TranslationJobCreator::operator()(Batch batch) const {
    return std::make_unique<TranslationJob>(_options, std::move(batch));
  }


  class ScoringJob : public BatchJob<ScoringResult> {
  public:
    ScoringJob(ScoringOptions options, Batch batch)
      : BatchJob(std::move(batch))
      , _options(std::move(options))
    {
    }

  protected:
    std::vector<ScoringResult> get_results(const Batch& batch) const override {
      spdlog::debug("Running batch scoring on {} examples", batch.num_examples());
      auto results = TranslatorPool::get_translator()->score_batch(batch.get_stream(0),
                                                                   batch.get_stream(1),
                                                                   _options);
      spdlog::debug("Finished batch scoring");
      return results;
    }

  private:
    const ScoringOptions _options;
  };

  TranslatorPool::ScoringJobCreator::ScoringJobCreator(ScoringOptions options)
    : _options(std::move(options))
  {
  }

  std::unique_ptr<BatchJob<ScoringResult>>
  TranslatorPool::ScoringJobCreator::operator()(Batch batch) const {
    return std::make_unique<ScoringJob>(_options, std::move(batch));
  }

}
