
# README for Planet Carbon Assessment App

## Planet Forest Carbon Project Demo
Author: Emma Reich

## Project Overview
This app demonstrates how Planet's Forest Carbon Diligence product can support forest carbon project developers in Brazil.
It shows an interactive visualization of carbon density and carbon gain/loss over time (2013–2023) and projects future carbon loss based on observed trends.

The goal is to highlight the value of Planet’s high-resolution data for identifying project areas, monitoring forest change, and quantifying carbon stocks.

## App Features
Raster Animation: View carbon density and carbon gain/loss maps year-by-year (2013–2023)

Interactive Timeseries Plot: Explore historical carbon stock trends and future projections

Summary Statistics: See key metrics relevant for forest carbon project planning

Narrative Summary: Ties visuals together to emphasize value proposition

## How to Launch the App
The app is hosted on Streamlit Cloud and can be accessed at:

https://planet-carbon-assessment.streamlit.app/

## Files Included
planet_carbon_app.py — Main application code

data — Raster layers for carbon density, canopy height, and canopy cover

requirements.txt — Python libraries needed to run the app

environment.yml — Dependancies

README.md — Project description

## Methods Summary

Raster Processing: Using rasterio to load, mask, and process TIFF files

Trend Analysis: Linear regression with scikit-learn to project future carbon accumulation

Visualization: Interactive maps and plots using matplotlib and plotly

Deployment: Hosted via Streamlit Community Cloud for easy access
