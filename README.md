# Retrieving Chest Radiograph Exams by Shared Findings and Report Similarity

This repository contains the code of the paper "Retrieving Chest Radiograph Exams by Shared Findings and Report Similarity". 
## Installation and configuration

Clone the repository with `git clone https://github.com/MinaAhmdian/retrieval.git` and install the packages torch, torchvision, transformers, tokenizers, huggingface_hub, hi-ml-multimodal, timm, open_clip_torch, peft, radgraph, requests, scikit-learn, numpy, pandas, pillow, pyyaml, joblib, duckdb, pydicom, albumentations, omegaconf, opencv-python-headless, tensorboard, sentencepiece, einops, fvcore, mup, tenacity, and yacs. The download of CheXpert Plus also needs redivis. The export of TAIX-Ray needs pyarrow. The MIMIC-CXR, PadChest and VinDr-CXR paths refer to the 512-pixel derivatives used in the study. Matching original releases alone does not guarantee the same decoded pixels. The CheXtriev population also uses a list of 12 training JPEGs unavailable from the original-image release. The configuration expects their release-relative paths, one per line. Obtain that study-specific list with authorized access before reproducing this comparison. The CheXtriev steps run in an environment with the packages of the CheXtriev repository.

Every path and parameter is set in `config.yaml`. Copy the configuration to a local file and replace `/PATH` with your folders. Pass the file path to each module. Every output is written under `paths.output_dir`. The statistics modules read the evidence files and summary tables named in `statistics.inputs`. CheXbert runs from its official code and checkpoint (`tools.chexbert`). CheXtriev runs from a checkout of its repository at the commit in `published.source`, adapted by `published/chextriev_adaptation.patch`. The weights of the public encoders are downloaded at the revisions in `retrieval.encoders`.


## Code structure

The code runs from the repository root, which is on the Python path. Each step is a class that takes the configuration and, where the step works on one dataset or one run, its name. Calling `run()` runs the step. For example, `Retrieval(Config("config.yaml"), "mimic").run()`, with `Retrieval` from `encoders/retrieval.py` and `Config` from `config.py`, evaluates the public encoders and the finding classifier on the MIMIC-CXR test set. A run name is `<backbone>__<configuration>__<seed>`. Every unit of work is written with a provenance record (`provenance.py`). An interrupted step continues where it stopped. A finished unit is reused only while its inputs and settings are unchanged.

Cohort, labels, and entities:

- `prep.chexpert_download.CheXpertDownload` downloads CheXpert Plus.
- `prep.taix_export.TaixExport` exports the images of TAIX-Ray.
- `corpora.vindr_headers.VinDrHeaders` reads the age, sex, and view of the VinDr-CXR images from their DICOM headers.
- `cohort.exam_builder.ExamBuilder` reads one dataset through its loader in `corpora/` and keeps up to 2 frontal and 2 lateral images per exam (`exam_selector.py`) with the official split or a patient split (`split_assigner.py`).
- `cohort.image_pass.ImagePass` computes the pixel hash of every image and the BioViL-T embeddings for the view classifier.
- `cohort.view_classifier.ViewClassifier` assigns a view to every image without a recorded view. The exam builder supplies the images for the image pass, whose features train the view classifier. Run the exam builder again with the assigned views before constructing the cohort. This second pass supplies the views for Open-I and the affected ReXGradient exams.
- `labels.chexbert_labeler.CheXbertLabeler` labels the findings and impression of every report.
- `labels.radgraph_annotator.RadGraphAnnotator` extracts the RadGraph-XL entities of the test reports.
- `cohort_files.evidence_table.EvidenceTable` joins every dataset into one table of exams and draws the test queries. It excludes the exams with an image whose pixels are identical to an image of an exam in an earlier split (`duplicate_filter.py`).
- `cohort_files.characteristics.Characteristics` computes the counts of every dataset and split and the agreement of CheXbert with the labeler output released with MIMIC-CXR and CheXpert Plus. The study cohort CSV has 1,035,129 exams after 364 duplicate exclusions. Its SHA-256 is b7b371bcaa43ef1870b00791d9b8e064c4b03d04e4a42e81d66c670a2098ce0e.
- `cohort_files.dataset_table.main_dataset_table` computes the exams, patients, test queries, and archive exams of each dataset and the shares of its exams with a lateral image, a context text, and age and sex. The overall patient count covers the eight datasets with patient identifiers. VinDr-CXR supplies image identifiers. The companion finding-support table gives positive query and archive counts and the PadChest missing-label bound for each eligible finding.

Public encoders and the finding classifier:

- `encoders.image_resizer.ImageResizer` resizes the images to 224 x 224 pixels.
- `encoders.model_fetcher.ModelFetcher` downloads the encoders. Their wrappers are in `encoders/models.py`.
- `encoders.image_embedder.ImageEmbedder` embeds the images.
- `encoders.text_embedder.TextEmbedder` embeds the reports and the context texts.
- `encoders.finding_classifier.FindingClassifier` fits one logistic regression per finding on frozen BioViL-T features.
- `encoders.retrieval.Retrieval` ranks the archive of one test set with every rule of the public encoders and with the profile rule of the classifier. It writes the per-query metrics (`query_metrics.py`, with `lib/metrics.py`).
- `encoders.encoder_files.EncoderFiles` joins the per-query metrics of all test sets into one evidence file and writes the mean over queries.
- `encoders.best_frozen.main_best_frozen` selects the best of the rules of the public encoders for each test set, archive, and metric.

Exam encoder:

- The model is `lib/exam_encoder.py` on a backbone with LoRA adapters (`trained_encoder/backbones.py`, `exam_model.py`). It is trained with the losses of `lib/losses.py` and evidence dropout. The ranking rules are in `lib/scoring.py`.
- `trained_encoder.slot_image_resizer.SlotImageResizer` resizes the images beyond the first frontal and the first lateral image.
- `trained_encoder.training_texts.TrainingTexts` embeds the reports of the training exams.
- `trained_encoder.context_permutations.ContextPermutations` draws the donor of the shuffled context.
- `trained_encoder.trainer.Trainer` trains one run.
- `trained_encoder.test_outputs.TestEmbedder` embeds the test exams with the chosen weights of a run.
- `trained_encoder.trained_retrieval.TrainedRetrieval` evaluates the profile, alignment, and two-stage rules on one test set.
- `trained_encoder.model_files.ModelFiles` joins the per-query metrics into one evidence file.

Depth of the two-stage rule, controls, transfer, and downstream use:

- `controls.validation` assigns and embeds the validation queries (`ValidationAssignments`, `ValidRadGraph`, `ValidEmbedder`, `ValidReportEmbedder`).
- `controls.controls_retrieval.ValidRetrieval` ranks the validation queries at each depth.
- `controls.depth_selection.DepthSelection` selects the depth with `lib/fusion.py`.
- `controls.permutations` draws the donors of the shuffled lateral and of the shuffled report.
- `controls.replacement_heads` fits the missing head of a model trained with one head on the training exams (`TrainEmbedder`, `FitReplacementHeads`).
- `controls.fusion_model.FusionModel` fuses the frontal and the lateral image inside the backbone.
- `controls.controls_retrieval.ControlsRetrieval` evaluates the controls, the shuffled report, and the two-stage rule at the selected depth on one test set.
- `controls.controls_files.ControlsFiles` writes the evidence file of the controls and their bootstrap summaries.
- `run_groups.group_retrieval.TransferRetrieval` evaluates the models trained without one training dataset.
- `run_groups.group_files.TransferFiles` writes the evidence file of these models and their bootstrap summaries.
- `downstream.downstream_predictions.DownstreamPredictions` predicts the findings of each query from the retrieved exams.
- `downstream.downstream_files.DownstreamFiles` computes the AUROC and F1 of these predictions with their bootstrap summaries (`downstream_metrics.py`).

CheXtriev:

- `published.population.PublishedPopulation` builds the image population of the CheXtriev protocol from MIMIC-CXR-JPG and the Chest ImaGenome scene graphs.
- `published.crops.PublishedCrops` writes the crops of the anatomical regions of each image.
- `published.chextriev` downloads the ResNet-50 weights (`FetchWeights`), trains CheXtriev (`ChextrievTrainer`), and embeds the test images (`ChextrievEmbedder`).
- `published.frontal` embeds each test image alone with the exam encoder (`FrontalImages`, `FrontalEmbedder`).
- `published.retrieval.PublishedRetrieval` computes the top-5 metrics of the protocol (`top5.py`).
- `published.study_common.StudyCommon` evaluates the methods on the common subset of MIMIC-CXR test exams.
- `published.files.PublishedFiles` writes the evidence file and the bootstrap summaries.

Reranking baselines and depth of the two-stage rule:

- `rerank.rerank_retrieval.RerankRetrieval` ranks one test set with reports in two stages at the selected depth. The first stage is the finding classifier or the profile of the exam encoder. The reranker is a rule of the public encoders or the alignment of the exam encoder. Besides the metrics of `query_metrics.py`, it writes the share of the ten highest ranked exams that are positive for each positive finding of the query (`share10`).
- `rerank.rerank_files.RerankFiles` writes the evidence file of these rankings and their bootstrap summaries. It checks that the two-stage rule of the exam encoder equals its evaluation in `controls.controls_retrieval.ControlsRetrieval`, and selects the best rule of the public encoders as reranker for each first stage, test set, archive, and metric.
- `rerank.rerank_retrieval.DepthRetrieval` evaluates the two-stage rule of the exam encoder and the finding classifier with its selected reranker for NDCG@10 at each depth of `depth.ks`.
- `rerank.rerank_files.DepthFiles` writes the evidence file of these rankings and their bootstrap summaries.

Differences and statistics (each `main_` function takes the path of `config.yaml`):

- `retrieval_levels.main_retrieval_levels(config_path, rankings, output_path)` computes individual retrieval scores, bootstrap standard deviations, and percentile intervals from per-query files. Each ranking specifies the evidence source (`encoders`, `model`, `controls`, or `transfer`), the unit fields listed in `retrieval_levels.UNIT`, and the training seeds. Use `["none"]` for a frozen encoder or the finding classifier. For the full encoder at depth 100, use the `controls` evidence with seeds `[42, 1337, 2024]`. Calling `main_retrieval_levels(config_path)` writes the full method and dataset grid to the statistics output prefix. The two CheXtriev evaluations use `published.files`.
- `patient_comparisons.main_patient_comparisons` resamples query patients for the two-stage ranking comparisons and adjusts the permutation tests across the classifier and all eligible frozen rules. Its output also gives each patient equal weight within a finding as a sensitivity analysis and gives simultaneous within-archive intervals for the frozen rules.
- `final_differences.main_final_differences` computes the paired differences between the exam encoder and every rule of the public encoders and the classifier, every control, the models trained without one dataset, and the queries without a lateral image or a context. Each difference has a bootstrap of the queries and a sign-flip permutation test with the Benjamini-Hochberg correction (`lib/stats_utils.py`) within its family.
- `top_ten_overlap.main_top_ten_overlap` computes abnormal-finding precision among the ten highest ranked exams and bounds the share that also reaches the report-entity Dice threshold. It computes paired differences across the full ranking and its ten highest results on the same query and finding pairs, with and without support devices.
- `paired_retrieval.main_paired_retrieval` compares the two-stage and profile rankings and tests input removal among queries with an available lateral view or history.
- `paired_retrieval.main_burden` computes query-independent finding-count and training-prevalence comparators in archives with reports.
- `secondary_patient.main_secondary_patient` computes clustered intervals for the matched CheXtriev comparison and for macro AUROC of the two ten-neighbor predictors.
- `patient_tests.main_patient_tests` resamples query patients for the remaining comparisons of the paper: the ablations, the baselines and single heads, the comparisons per finding, transfer to an omitted source, and the CheXtriev protocol. It also gives patient-level intervals for the NDCG@10 of the image-only archives.
- `extra_differences.main_extra_differences` computes the paired differences for the copy of the frontal, the shuffled context, and the queries without a lateral image or a context under the profile and alignment rules. It also computes the comparisons with CheXtriev on its protocol and on the common subset.
- `rule_differences.main_rule_differences` compares the finding classifier with every rule of the public encoders and the ranking rules of the exam encoder with each other.
- `downstream.downstream_differences.main_downstream_differences` compares the predictions from the exams retrieved by the exam encoder with those of the finding classifier.
- `rerank_tests.main_rerank_tests` resamples query patients for the comparisons of the two-stage rule of the exam encoder with the best reranker of the finding classifier, the best reranker of the profile, and the finding classifier reranked by the alignment of the exam encoder. The p values of all these comparisons form one Benjamini-Hochberg family. For NDCG@10 it gives simultaneous intervals over the rules of the public encoders.
- `seed_variation.main_seed_variation` computes the standard deviation over the training seeds from the summaries with one row per seed.

