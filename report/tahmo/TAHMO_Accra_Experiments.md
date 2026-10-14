# TAHMO weather stations in Accra: experiments

Generated 2026-10-14 18:12 UTC by `scripts/12_tahmo_accra.py experiments`.

## E0. Stations and coverage

Daily values need 75% of the 5-minute samples (00-24 UTC). Columns `days_*` count valid days per variable.

| code | name | lat | lon | km_from_centre | first | last | days_t2m | days_t2m_range | days_rh | days_precip | days_ws10 | days_ps | days_ssrd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TA00016 | Accra Academy School, Accra | 5.573 | -0.245 | 6.7 | 2024-08-01 | 2026-02-28 | 380 | 380 | 380 | 0 | 374 | 380 | 380 |
| TA00098 | ARCUS- GMet HQ | 5.652 | -0.164 | 6.4 | 2024-08-01 | 2026-10-14 | 466 | 466 | 466 | 352 | 458 | 425 | 466 |
| TA00118 | Tema Secondary School | 5.641 | -0.012 | 20.2 | 2025-05-22 | 2026-06-17 | 282 | 282 | 282 | 0 | 275 | 282 | 282 |
| TA00127 | St. John's Grammar | 5.638 | -0.245 | 7.4 | 2024-08-01 | 2026-02-27 | 506 | 506 | 506 | 503 | 484 | 506 | 506 |
| TA00313 | Safisana | 5.684 | -0.049 | 18.1 | 2024-08-01 | 2026-10-14 | 803 | 803 | 803 | 456 | 776 | 803 | 803 |
| TA00567 | ARCUS- Accra Girls SHS | 5.597 | -0.194 | 0.6 | 2024-08-01 | 2026-10-14 | 775 | 775 | 775 | 770 | 766 | 775 | 775 |
| TA00647 | UG Greenhouse, Legon | 5.66 | -0.19 | 6.6 | 2024-08-01 | 2026-10-14 | 673 | 673 | 673 | 663 | 655 | 673 | 673 |
| TA00885 | Dutch Embassy | 5.574 | -0.191 | 2.9 | 2024-08-01 | 2026-10-14 | 622 | 622 | 622 | 622 | 622 | 622 | 622 |

## E1. How far the model's weather input (MERRA-2 via NASA POWER) is from the ground in Accra

`merra_*` compares MERRA-2, bilinearly interpolated to each TAHMO station, with the station's daily value. `corrected_*_cv` is a linear correction fitted on other months (month-blocked CV). TAHMO wind is measured at about 2 m, MERRA-2 wind at 10 m, so a wind bias is expected.

| variable | n | tahmo_mean | merra_mean | merra_bias | merra_rmse | merra_r | corrected_rmse_cv | corrected_r_cv |
|---|---|---|---|---|---|---|---|---|
| t2m | 4499 | 27.384 | 26.384 | -1.0 | 1.264 | 0.884 | 0.761 | 0.883 |
| t2m_range | 4499 | 6.547 | 4.34 | -2.207 | 2.949 | 0.324 | 1.944 | 0.233 |
| rh | 4499 | 88.596 | 86.059 | -2.536 | 5.758 | 0.464 | 5.012 | 0.454 |
| precip | 3360 | 2.468 | 4.507 | 2.039 | 13.661 | 0.229 | 13.339 | 0.196 |
| ws10 | 4402 | 1.161 | 3.108 | 1.947 | 2.041 | 0.454 | 0.332 | 0.445 |
| ps | 4458 | 100.682 | 100.421 | -0.261 | 0.369 | 0.656 | 0.262 | 0.646 |
| ssrd | 4484 | 3.46 | 4.741 | 1.281 | 1.752 | 0.599 | 1.197 | 0.595 |

## E2. Low-cost sensor calibration: model humidity versus measured humidity

Pairs are low-cost sensors within 10 km of the US Embassy BAM-1020 on the same day, restricted to days where both humidity sources exist. Errors are month-blocked cross-validated, in µg/m³ against the BAM.

| network | humidity | correction | n_pairs | n_sensors | rh_range | n | mean_obs | bias | rmse | mae | r | r2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AirGradient | merra | ratio | 471 | 4 | 76-94 | 471 | 19.197 | -0.27 | 3.844 | 2.641 | 0.907 | 0.822 |
| AirGradient | merra | loglinear | 471 | 4 | 76-94 | 471 | 19.197 | -0.694 | 4.207 | 2.767 | 0.896 | 0.786 |
| AirGradient | tahmo | ratio | 471 | 4 | 79-100 | 471 | 19.197 | -0.27 | 3.844 | 2.641 | 0.907 | 0.822 |
| AirGradient | tahmo | loglinear | 471 | 4 | 79-100 | 471 | 19.197 | -0.553 | 3.816 | 2.442 | 0.915 | 0.824 |
| AirGradient | none | raw sensor | 471 | 4 |  | 471 | 19.197 | 4.14 | 5.767 | 4.717 | 0.914 | 0.598 |
| Clarity | merra | ratio | 746 | 29 | 67-94 | 746 | 24.146 | 0.226 | 8.498 | 6.303 | 0.695 | 0.281 |
| Clarity | merra | loglinear | 746 | 29 | 67-94 | 746 | 24.146 | -0.714 | 7.323 | 5.54 | 0.693 | 0.466 |
| Clarity | tahmo | ratio | 746 | 29 | 68-98 | 746 | 24.146 | 0.226 | 8.498 | 6.303 | 0.695 | 0.281 |
| Clarity | tahmo | loglinear | 746 | 29 | 68-98 | 746 | 24.146 | -1.031 | 7.99 | 6.069 | 0.618 | 0.365 |
| Clarity | none | raw sensor | 746 | 29 |  | 746 | 24.146 | 4.302 | 10.509 | 7.898 | 0.695 | -0.099 |
| all low-cost | merra | ratio | 1217 | 33 | 67-94 | 1217 | 22.231 | -0.265 | 6.957 | 4.858 | 0.763 | 0.513 |
| all low-cost | merra | loglinear | 1217 | 33 | 67-94 | 1217 | 22.231 | -0.743 | 6.463 | 4.635 | 0.767 | 0.58 |
| all low-cost | tahmo | ratio | 1217 | 33 | 68-100 | 1217 | 22.231 | -0.265 | 6.957 | 4.858 | 0.763 | 0.513 |
| all low-cost | tahmo | loglinear | 1217 | 33 | 68-100 | 1217 | 22.231 | -0.569 | 6.68 | 4.658 | 0.747 | 0.551 |
| all low-cost | none | raw sensor | 1217 | 33 |  | 1217 | 22.231 | 4.239 | 8.976 | 6.667 | 0.769 | 0.19 |

## E3. Does local measured weather improve GH-PM25 in Accra?

Accra station-days (sites within 30 km of the centre and 15 km of a TAHMO station). Weather B replaces MERRA-2 temperature, humidity, rain and wind (and their lags) with TAHMO values mapped to the MERRA-2 scale by the E1 regression. Leave-month-out lets the model learn from Accra in other months; leave-Accra-out never shows it Accra.

| design | weather | learners | n | mean_obs | bias | rmse | mae | r | r2 |
|---|---|---|---|---|---|---|---|---|---|
| leave-month-out (Accra) | MERRA-2 (current) | catboost+extratrees | 8017 | 17.611 | 0.048 | 4.286 | 1.938 | 0.842 | 0.626 |
| leave-month-out (Accra) | TAHMO, mapped to MERRA-2 scale | catboost+extratrees | 8017 | 17.611 | -0.369 | 3.874 | 2.167 | 0.835 | 0.695 |
| leave-Accra-out | MERRA-2 (current) | catboost+extratrees | 8017 | 17.611 | 0.163 | 4.916 | 2.608 | 0.799 | 0.508 |
| leave-Accra-out | TAHMO, mapped to MERRA-2 scale | catboost+extratrees | 8017 | 17.611 | -0.069 | 4.975 | 3.349 | 0.705 | 0.496 |
