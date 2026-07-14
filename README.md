# Delhi Air Quality Prediction & Personalized Health Risk Advisor

## Overview
This project focuses on forecasting the Air Quality Index (AQI) for the Delhi/NCR region and translating those predictions into actionable, personalized health advisories. 

While most air quality models stop at predicting a raw AQI number, this project introduces a human-centric layer. It evaluates the forecasted pollution levels against specific user health profiles (e.g., parents of young children, asthma patients, the elderly, outdoor workers, and healthy adults) to provide targeted, risk-aware guidance rather than generic warnings. 

This project is built with a focus on real-world civic impact, aligning with e-governance themes by addressing one of the most critical public health challenges in the region.

## Key Features
* **AQI Forecasting:** Analyzes historical pollution patterns alongside meteorological data to predict upcoming AQI levels.
* **Personalized Health Advisory Layer:** A dynamic recommendation engine that maps predicted AQI ranges to specific demographic and health categories.
* **Actionable Insights:** Translates raw environmental data into specific guidance (e.g., "Avoid outdoor activity after 4 PM; expected AQI > 350 is particularly risky for asthma profiles").

## Data Strategy & Sources
To ensure high credibility and predictive accuracy, this project avoids unverified crowdsourced datasets in favor of official government monitoring records and real-time APIs. The data pipeline is split into historical training data and live inference data:

* **Historical AQI Data (Training):** Sourced from official CPCB/DPCC government monitoring stations via [OpenCity.in](https://data.opencity.in/dataset/delhi-hourly-air-quality-reports). The primary dataset utilizes comprehensive hourly records (2017-2023) from the Anand Vihar station.
* **Historical Weather Data (Training):** Sourced via the [OpenWeatherMap API](https://openweathermap.org/api) to capture temperature, humidity, and wind speed, which are critical meteorological features that directly influence particulate matter concentration.
* **Live AQI Data (Inference):** Real-time integration via the [World Air Quality Index (WAQI) API](https://aqicn.org/data-platform/token/) to provide current pollution metrics for the live application.
* **Live Weather Data (Inference):** Real-time meteorological data fetched via OpenWeatherMap to feed current conditions into the forecasting model.

## Current Project Status
This repository currently contains the initial foundational notebooks for data handling, cleaning, and exploratory data analysis (EDA).
