---

# Model Card for {{ model_id | default("Model ID", true) }}

<!-- Provide the model name and a 1-2 sentence summary of what the model is. -->

{{ model_summary | default("", true) }}

## Model Details

### Model Description

<!-- Provide basic details about the model. This includes the architecture, version, if it was introduced in a paper, if an original implementation is available, and the creators. Any copyright should be attributed here. General information about training procedures, parameters, and important disclaimers can also be mentioned in this section. -->

{{ model_description | default("", true) }}

<!-- List (and ideally link to) the people who built the model. -->
- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
<!-- List (and ideally link to) the funding sources that financially, computationally, or otherwise supported or enabled this model. -->
- **Funded by [optional]:** {{ funded_by | default("[More Information Needed]", true)}}
<!-- Supervision/learning method, machine learning type, and modality -->
- **Model type:** {{ model_type | default("[More Information Needed]", true)}}
<!-- Kumar Lab models are typically CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/) -->
- **License:** {{ license | default("[More Information Needed]", true)}}
<!-- If this model has another model as its base, link to that model here. -->
- **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}}

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

<!-- This section addresses questions around how the model is intended to be used in different applied contexts, discusses the foreseeable users of the model (including those affected by the model), and describes uses that are considered out of scope or misuse of the model. Note this section is not intended to include the license usage details. For that, link directly to the license. -->

### Direct Use

<!-- Explain how the model can be used without fine-tuning, post-processing, or plugging into a pipeline. An example code snippet is recommended. -->

{{ direct_use | default("[More Information Needed]", true)}}

### Downstream Use [optional]

<!-- Explain how this model can be used when fine-tuned for a task or when plugged into a larger ecosystem or app. An example code snippet is recommended. -->

{{ downstream_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

<!-- List how the model may foreseeably be misused (used in a way it will not work for) and address what users ought not do with the model. -->

{{ out_of_scope_use | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

<!-- This section identifies foreseeable harms, misunderstandings, and technical and sociotechnical limitations. It also provides information on warnings and potential mitigations. Bias, risks, and limitations can sometimes be inseparable/refer to the same issues. Generally, bias and risks are sociotechnical, while limitations are technical: -->
<!-- A bias is a stereotype or disproportionate performance (skew) for some subpopulations. -->
<!-- A risk is a socially relevant issue that the might cause. -->
<!-- A limitation is a likely feature failure model that can be addressed following the listed Recommendations. -->

{{ bias_risks_limitations | default("[More Information Needed]", true)}}

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

{{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}}

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

<!-- This section provides information to describe and replicate training, including the training data, the speed and size of training elements, and the environmental impact of training. This relates heavily to the Technical Specifications as well, and content here should link to that section when it is relevant to the training procedure. It is useful for people who want to learn more about the model inputs and training footprint. It is relevant for anyone who wants to know the basics of what the model is learning. -->

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

<!-- Detail tokenization, resizing/rewriting (depending on the modality), etc. -->

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols, what is being measured in the evaluation, and provides the results. Evaluation is ideally constructed with factors, such as domain and demographic subgroup, and metrics, such as accuracy, which are prioritized in light of foreseeable error contexts and groups. Target fairness metrics should be decided based on which errors are more likely to be problematic in light of the model use. You can also specify your model’s evaluation results in a structured way in the model card metadata. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

<!-- Results should be based on the Factors and Metrics defined above. -->

{{ results | default("[More Information Needed]", true)}}

#### Summary

<!-- What do the results say? This can function as a kind of tl;dr for general audiences. -->

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Model Card Authors [optional]

{{ model_card_authors | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}