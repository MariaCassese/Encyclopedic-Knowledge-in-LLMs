This repository contains the code and experimental framework associated with the paper:

**Benchmarking Accuracy and Bias on Encyclopedic Knowledge in (Vision-)Language Models**  
Maria Cassese, Giovanni Puccetti, Andrea Esuli  
*Expert Systems With Applications*, 2027

The benchmark is built from Wikidata and contains **108,854 entities** across six categories:

- Country Leaders
- Actors
- Film Directors
- Olympic Athletes
- Literary Characters
- Trademarks

Each entity is enriched with metadata such as geographic information, temporal information, gender, and visual content when available.

The framework evaluates models using controlled prompt templates with different combinations of information, including:

- entity name
- country
- dates
- profession
- image
- literary work

The goal is not only to measure overall factual accuracy, but also to investigate whether model performance is systematically associated with:

- geographic region
- gender
- online presence
- socioeconomic and development indicators
- language

The experiments include both small and large open-weight models, vision-language models, mixture-of-experts architectures, and a proprietary model.
