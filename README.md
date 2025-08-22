# Nintendo Games Dashboard

An [interactive **Streamlit dashboard**](https://nintendo.streamlit.app) providing insights into more than **22,000 Nintendo games**.  
The project was developed as part of a **3-day group challenge** in the Data Science & AI Bootcamp (Batch 32) at [Constructor Academy](https://academy.constructor.org).  

-   Built with **Python, Plotly, and Streamlit**  
-   Data collected via **web scraping from [DekuDeals](https://www.dekudeals.com/)**  
-   Designed for both **Gamers** and **Developers**  

---

## Features

-   **Gamers' View** (by [Damla](https://github.com/damlinaa))  
    - Search/filter by price, review scores, discounts  
    - Quickly discover the best-value games  

-   **Developers' View** (by [Debbie](https://github.com/Debbcwing))
    - Analyze game success factors (genres, publishers, release trends)  
    - Identify market trends and opportunities  

-   **Data Foundation** (by [Karlo](https://github.com/karlolukic))  
    - Scraped ~22,000 Nintendo games from DekuDeals  
    - Attributes include: title, release date, price, publisher, review scores  
    - Cleaned and transformed into structured dataset with Python/Pandas  

-   Fully interactive visualizations built with **Plotly**  
-   Deployed as a **Streamlit web app**  

---

## Repository Structure

```text
nintendo-games-dashboard/
├── data/                          # scraped datasets (CSV)
├── Welcome.py/                    # Streamlit dashboard entry point
├── pages/
│   ├── 01_Gamers.py               # gamers page in Streamlit dashboard
│   ├── 02_Developers.py           # developers page in Streamlit dashboard
├── requirements.txt               # dependencies
├── slides/                        # PowerPoint presentation
└── README.md
```

---

## Data Source

-   **Nintendo game data**: [DekuDeals](https://www.dekudeals.com)  

Thanks to DekuDeals for making comprehensive Nintendo game data accessible.

---

## Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/nintendo-challenge/nintendo-games-dashboard.git
cd nintendo-games-dashboard
pip install -r requirements.txt
```

---

## Run the Streamlit App

```bash
streamlit run scripts/Welcome.py
```

The app will open in your browser.

---

## Contributors

This dashboard was developed collaboratively during a bootcamp group challenge:  

-   **Damla** – Gamers' Insights UI  
-   **Debbie** – Developers' Insights UI  
-   **Karlo** – Web Scraping & Data Collection  

---

## Acknowledgements

This project was built as part of the **Data Science & AI Bootcamp (Batch 32)** at [Constructor Academy](https://academy.constructor.org).  
Special thanks to the instructors and fellow participants for guidance and feedback throughout the project.
