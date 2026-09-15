# Literature notes (compiled for the technical report)

## Baseline: Anand et al. (2026), EarthArXiv, doi:10.31223/X5KR3D
- 0.01 deg daily PM2.5, Ghana, 2005-2025; dataset zenodo.org/records/19636051
- 98 monitors (96 LCS: Airnote, Clarity, PurpleAir, QuantAQ; 2 BAM-1020: US Embassy Accra, Techiman), Aug 2018 - Oct 2024
- Inputs: MAIAC AOD, OMI/TROPOMI L2 (UVAI, O3, NO2, SO2, HCHO, CO), ERA5-Land, MERRA-2 aerosol species, DOY sin/cos.
  No land use, population, roads, elevation, fire, PBLH, or lags.
- Complete-case training; 3 sub-models (OMI/TROPOMI/MERRA-2)
- XGBoost best. Random CV R2 0.72 (RMSE 12.5); temporal (LOYO) R2 0.56-0.59; spatial (LOLO) R2 0.49-0.57 (RMSE 16.8-18.9)
- Hyper-parameters tuned on random CV; slope 0.72 (high-end under-prediction); no uncertainty; no AOA.
- SHAP: RH > DOY > U/V wind > precipitation; UVAI positive.
- North (>8N) 1.63-1.83x south; Harmattan mode ~69 vs ~20 ug/m3.
- Predecessor: Westervelt, Amooli, Anand (2025) ACS ES&T Air 2(8):1468, doi:10.1021/acsestair.4c00366

## Methods references
- Wei et al. 2020 ACP 20:3273 (STET, CV R2 0.89 / out-of-station 0.88); Wei et al. 2021 RSE 252:112136; Wei et al. 2023 Nat Commun 14:8349
- Di et al. 2019 Environ Int 130:104909 (NN+RF+GBM ensemble, GAM, R2 0.86)
- Hammer et al. 2020 ES&T 54:7879; van Donkelaar et al. 2021 ES&T 55:15287; Shen et al. 2024 ACS ES&T Air 1:332
- Shtein et al. 2020 ES&T 54:120 (Italy ensemble); Just et al. 2020 Atmos Env 239:117649 (spatial CV RMSE 48% worse than random)
- Hengl et al. 2018 PeerJ 6:e5518 (RFsp); Meyer & Pebesma 2021 MEE 12:1620 (AOA); Meyer & Pebesma 2022 Nat Commun 13:2208;
  Ploton et al. 2020 Nat Commun 11:4540; Wadoux et al. 2021 Ecol Model 457:109692
- Lei et al. 2018 JASA 113:1094 (conformal); Meinshausen 2006 JMLR 7:983 (QRF); Mao, Martin & Reich 2024 JASA 119:904
- Adjei et al. 2026 arXiv:2604.22787 (African conformal PM2.5; spatial CV R2 0.13)
- LCS calibration: McFarlane et al. 2021 ACS ESC 5:2268 (GMR Accra R2 0.88); Raheja et al. 2023 ES&T 57:10708;
  Raheja et al. 2022 ACS ESC 6:1011 (Lome); Adong et al. 2022 Applied AI Letters 3:e76 (AirQo); Barkjohn et al. 2021 AMT 14:4617
- Context: Alli et al. 2021 ERL 16:074013 (Accra, Harmattan 89 vs 23); Alli et al. 2023 STOTEN 875:162582;
  Ofosu et al. 2013 JAWMA 63:1036 (Navrongo); Adhvaryu et al. 2024 Econ J (Harmattan mortality)
- CAMS: Inness et al. 2019 ACP 19:3515; Gueymard & Yang 2020 Atmos Env 225:117216; Garrigues et al. 2022 ACP 22:14657;
  Jin et al. 2022 Atmos Env 274:118972
- Standards: WHO 2021 AQG (24-h 15; annual 5; IT 75/50/37.5/25); Ghana EPA GS 1236:2019 24-h 35;
  US EPA AQI 2024 breakpoints 9.0/35.4/55.4/125.4/225.4
