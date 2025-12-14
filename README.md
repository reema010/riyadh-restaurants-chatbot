# Riyadh Restaurants Chatbot (RFQH)

A data-driven chatbot that answers questions about restaurants in Riyadh using a cleaned CSV dataset.
The chatbot supports filtering by neighborhood/area, cuisine/category, rating threshold, price range hints, and nearest results using latitude and longitude.

## Project Structure

* `app_fixed.py` — FastAPI backend (chatbot engine + API endpoints)
* `index_fixed.html` — Simple web-based user interface
* `data/riyadh_restaurants_clean_exported.csv` — Cleaned restaurant dataset

## Data Preparation
The dataset was cleaned and preprocessed using a dedicated Python script located in `scripts/data_cleaning.py`.


## How to Run

### 1) (Optional) Create and activate a virtual environment

```bash
conda create -n rfqh python=3.10 -y
conda activate rfqh
```

### 2) Install requirements

```bash
pip install -r requirements.txt
```

### 3) Set dataset path (PowerShell)

```powershell
$env:CSV_PATH="data\riyadh_restaurants_clean_exported.csv"
```

### 4) Run the API

```bash
uvicorn app_fixed:app --reload --reload-dir .
```

### 5) Test the application

* Health check:

  * `http://127.0.0.1:8000/health`
* Chat endpoint:

  * `POST http://127.0.0.1:8000/chat`
* Web interface:

  * Open `index_fixed.html` in a browser

## Example Questions

* اعطني أفضل 5 مطاعم تقييماً في حي الياسمين
* أقرب كافيه لي؟ موقعي 24.744, 46.636
* أبغى مطاعم لبنانية سعرها متوسط وتقييمها فوق 4
* وين المطاعم اللي تصنيفها Fine Dining؟

## Technologies Used

* Python
* Pandas (data loading and cleaning)
* FastAPI + Uvicorn (REST API backend)
* HTML + JavaScript (simple frontend interface)

## How the Chatbot Works

1. Load and clean the CSV dataset (remove duplicates, handle missing values, normalize text fields).
2. Parse the user query using a custom NLP pipeline to extract:

   * Number of results (top-K)
   * Area/neighborhood
   * Cuisine/category
   * Rating threshold
   * Price hints
   * Latitude/longitude for nearest-restaurant queries
3. Filter the dataset based on extracted criteria.
4. Rank results by rating or distance (for nearest queries).
5. Return a response grounded entirely in the restaurant data.

## Optional LLM Enhancement

The system supports an optional LLM + RAG layer using the OpenAI API and LangChain.
This layer is used only to improve response phrasing and clarity, while all results remain grounded in the dataset.

To enable it:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:USE_LLM_RAG="1"
```
