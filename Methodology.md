# Chatbot Methodology

This document explains how the data is processed, how the chatbot is connected to the dataset, and the rationale behind the chosen technologies.

---

## 1. Data Processing

The restaurant data is provided as a CSV file. Before using it in the chatbot, a preprocessing stage is applied to ensure data quality and consistency.

The following steps are performed:

* Import the CSV file using Pandas.
* Remove duplicate rows to avoid repeated results.
* Handle missing values by dropping records that lack essential fields such as restaurant name or geographic coordinates.
* Convert numeric fields (rating, latitude, longitude) to the correct data types.
* Normalize text fields (name, category, address, price) by trimming spaces and converting them to lowercase to support reliable text matching.

The cleaned data is stored in a Pandas DataFrame, which allows efficient filtering and querying during chatbot interaction.

---

## 2. Chatbot–Data Integration

The chatbot is directly connected to the processed DataFrame and retrieves answers strictly from the dataset.

The interaction flow is as follows:

1. The user submits a question through the web interface or API endpoint.
2. A custom NLP pipeline parses the question to extract relevant parameters such as:

   * Area or neighborhood
   * Cuisine or restaurant category
   * Rating threshold
   * Price hints
   * Number of results requested (top-K)
   * Geographic coordinates (latitude and longitude) for nearest-restaurant queries
3. These extracted parameters are used to filter the DataFrame.
4. Results are ranked logically, either by rating or by distance when location coordinates are provided.
5. The chatbot returns a response generated entirely from the filtered dataset, ensuring that answers are data-driven and not generic.

This design guarantees that every response reflects real restaurant data rather than free-form text generation.

---

## 3. Technology Selection Rationale

The following technologies were chosen for specific reasons:

* **Python**: Provides a rich ecosystem for data processing, backend development, and rapid prototyping.
* **Pandas**: Used for loading, cleaning, and filtering the restaurant dataset efficiently.
* **FastAPI**: Enables building a lightweight, fast, and well-structured REST API for chatbot communication.
* **HTML and JavaScript**: Used to implement a simple and accessible user interface for interacting with the chatbot.
* **Custom NLP Pipeline**: Chosen to maintain full control over how user queries are interpreted and mapped to dataset filters, ensuring transparency and correctness.

### Optional LLM Enhancement

The system also supports an optional Large Language Model (LLM) layer using the OpenAI API and LangChain. This component is designed to enhance response clarity and natural language phrasing only.

The core logic and decision-making remain fully data-driven, and the LLM does not generate answers independently of the dataset. This hybrid approach balances explainability, accuracy, and improved user experience.

---

This architecture ensures that the chatbot is reliable, interpretable, and suitable for practical use cases involving structured restaurant data.
