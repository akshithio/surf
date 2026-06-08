# Data & Benchmarks

This is the canonical data and benchmark note for SURF. It folds the old `data.md` and `benchmarks.md` into one file so dataset choice, access links, evaluation targets, and data-shape requirements stay in the same place.

Last source refresh: 2026-05-25.

## Bottom Line

CropHarvest is a useful prototype and strict-transfer benchmark, but it is not enough data or input structure for the stronger foundation-model version of the project.

The current data direction is:

```text
SSL4EO-S12 v1.1 + climate/terrain + agriculture-focused sampling -> patch-time-series pretraining
CropHarvest v2 + one external crop benchmark -> strict transfer evaluation
```

For `[7]`, keep the channel contract deliberately simple and matched:

```text
S2:      B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI
S1:      VV, VH
Context: temperature, precipitation, elevation
```

Do not include `crop_prior` in `[7]`. Add it only after the same external crop-confidence prior is available for both SSL4EO pretraining samples and CropHarvest evaluation points.

## What The Model Needs From Data

| Need | Why it matters | Current status | Data direction |
|---|---|---|---|
| Unlabeled crop-relevant scale | CropHarvest point pretraining saturates quickly and cannot define the ceiling. | Weak | SSL4EO-S12 v1.1 first; custom Sentinel chips later. |
| Spatial context | `N x T x C x 1 x 1` cannot learn field boundaries, mixed pixels, texture, or within-field heterogeneity. | Missing | SSL4EO-S12 patches, PASTIS, ZueriCrop, raw Sentinel chips. |
| Actual acquisition timing | Synthetic monthly timing cannot represent real revisit gaps or cloud-induced irregularity. | Weak | Raw STAC Sentinel pulls; dataset metadata where available. |
| Cloud and quality masks | Robustness should use real missingness, not only synthetic deletion. | Partial | SSL4EO-S12 cloud masks; Sentinel-2 SCL/QA; CloudSEN12 if needed. |
| Crop-focused sampling | Generic EO pretraining wastes capacity on forests, cities, water, and bare land. | Missing | WorldCereal, ESA WorldCover, Dynamic World, USDA CDL. |
| Climate and phenology | Crop dynamics depend on heat, precipitation, radiation, water stress, and season. | Minimal | AgERA5, ERA5-Land, Daymet, CHIRPS. |
| Soil and terrain | Yield/stress transfer can be dominated by static field context. | Partial | SRTM, SoilGrids, HWSD, SMAP. |
| External crop labels | Need to show transfer beyond CropHarvest. | Missing | EuroCropsML, TimeSen2Crop, ZueriCrop, BreizhCrops, PASTIS. |
| Stress/yield labels | Needed for any honest stress/yield claim. | Missing | YieldSAT, CropNet, SustainBench, Agriculture-Vision, USDA QuickStats/CDL. |

## Immediate Priority

| Priority | Source | What it is | Shape / scale | Best use | Limitation | Links |
|---:|---|---|---|---|---|---|
| 0 | CropHarvest | Global crop/non-crop and crop-type benchmark with Sentinel-1, Sentinel-2, ERA5, and DEM-derived point time series. | About 95k datapoints; about 70k paired feature samples; 12-month point sequences. | Strict heldout evaluation and fast prototype dataset. Build `cropharvest_v2.zarr` with the `[7]` contract. | Too small and too point-level for main foundation pretraining. | [paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/54229abfcfa5649e7003b83dd4755294-Paper-round2.pdf), [Zenodo](https://zenodo.org/records/5037916), [GitHub](https://github.com/nasaharvest/cropharvest), [bands.py](https://raw.githubusercontent.com/nasaharvest/cropharvest/main/cropharvest/bands.py) |
| 0 | SSL4EO-S12 v1.1 | Multimodal, multiseasonal Sentinel pretraining dataset. | 246,144 locations with four timestamps; S2 L1C/L2A/RGB, S1 GRD, land cover, DEM, NDVI, cloud masks; nearly 1M image patches. | First real patch-time-series pretraining corpus for `[7]`. | Only four seasonal timestamps; not agriculture-filtered by default; climate must be aligned separately. | [Hugging Face](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1), [paper](https://arxiv.org/abs/2503.00168), [code](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1) |
| 0 | AgERA5 | Daily agriculture-oriented reanalysis indicators derived from ERA5. | Global, daily, from 1979 onward; agriculturally relevant temperature, precipitation, vapor pressure, radiation, and related indicators. | Climate side channel, GDD, drought/water-stress summaries. | Coarse relative to 10m imagery; must be cached carefully. | [CDS](https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators), [ECMWF docs](https://confluence.ecmwf.int/pages/viewpage.action?pageId=278551004) |
| 0 | ERA5-Land | Hourly land reanalysis. | Global, about 9km grid spacing, hourly, from 1950 to near-present. | Backup/default climate source if AgERA5 is inconvenient. | Requires aggregation and feature engineering. | [ECMWF](https://www.ecmwf.int/en/era5-land), [GEE hourly](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY), [GEE daily](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR) |
| 0 | ESA WorldCover | Global land-cover map. | 10m global product for 2021; 11 classes including cropland. | Fast cropland mask and negative sampling control. | Coarse class taxonomy; not crop-type ground truth. | [WorldCover 2021](https://worldcover2021.esa.int/) |
| 0 | WorldCereal | Global crop extent and seasonal crop products. | 10m global maps; 2021 product collection. | Agriculture-filter SSL4EO/raw Sentinel samples; later crop-prior ablation. | Coarse crop taxonomy. | [products](https://esa-worldcereal.org/en/products/global-maps), [about](https://esa-worldcereal.org/en/about-worldcereal) |
| 0 | Dynamic World | Near-real-time global land-use/land-cover probabilities. | 10m Sentinel-2-derived class probabilities and labels for 9 classes. | Probabilistic cropland context and temporal land-cover confidence. | Derived from Sentinel-2; optical/cloud biases matter. | [about](https://dynamicworld.app/about/), [GEE](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) |

## External Crop Benchmarks

Use these to keep the project from becoming only a CropHarvest story.

| Priority | Source | What it is | Shape / scale | Best use | Limitation | Links |
|---:|---|---|---|---|---|---|
| 1 | EuroCropsML | Ready-to-use few-shot crop type benchmark built from EuroCrops reference data and Sentinel-2 L1C. | 706,683 parcel-level datapoints, 176 classes, annual 2021 Sentinel-2 time series over Estonia, Latvia, and Portugal. | Best first external crop-type benchmark. | Europe-only; parcel-median time series, not dense patches. | [docs](https://eurocropsml.readthedocs.io/), [paper](https://arxiv.org/abs/2407.17458), [Zenodo](https://zenodo.org/records/12168505) |
| 1 | TimeSen2Crop | Large Sentinel-2 crop-type time-series dataset for Austria. | More than 1M pixel samples, 16 crop types, one agronomic year. | Pixel-level Sentinel-2 crop-type transfer. | Single country; Sentinel-2 only. | [Zenodo](https://zenodo.org/records/4715631), [paper page](https://research.utwente.nl/en/publications/timesen2crop-a-million-labeled-samples-dataset-of-sentinel-2-imag) |
| 1 | ZueriCrop | Swiss Sentinel-2 crop mapping benchmark. | 28k Sentinel-2 patches of size `24 x 24`, 71 observations over 52 weeks, 48 classes, 116k field instances. | Spatial patch time-series transfer. | Switzerland-only. | [paper](https://www.sciencedirect.com/science/article/pii/S0034425721003230), [arXiv](https://arxiv.org/abs/2304.11456) |
| 1 | BreizhCrops | Parcel-level Sentinel-2 time series for crop type mapping in Brittany, France. | About 580k field parcels over one year. | Large crop-type time-series benchmark. | Single region and mostly parcel-level. | [site](https://breizhcrops.org/), [paper](https://arxiv.org/abs/1905.11893) |
| 1 | PASTIS / PASTIS-HD | Agricultural parcel semantic/panoptic segmentation from satellite time series. | 2,433 one-square-kilometer patches with Sentinel-2 time series; PASTIS-HD adds aligned high-resolution imagery and Sentinel-1/2 variants. | Dense spatial outputs or segmentation evidence. | Smaller and France-focused. | [Zenodo](https://zenodo.org/records/5012942), [paper](https://arxiv.org/abs/2107.07933), [GFM-Bench mirror](https://huggingface.co/datasets/GFM-Bench/PASTIS), [PASTIS-HD](https://huggingface.co/datasets/IGNF/PASTIS-HD) |
| 1 | Sen4AgriNet / AgriSen-COG | Multi-country, multi-temporal Sentinel-2 crop classification/segmentation datasets derived from European LPIS-style declarations. | European multi-country crop mapping; AgriSen-COG includes Austria, Belgium, Spain, Denmark, and the Netherlands. | Domain adaptation across countries and crop taxonomies. | Europe-heavy; preprocessing work likely. | [Sen4AgriNet](https://arxiv.org/abs/2204.00951), [AgriSen-COG](https://www.mdpi.com/2332078) |

## Raw Imagery Access

SSL4EO-S12 v1.1 is the fastest bridge. A custom extraction pipeline is still needed for longer irregular crop seasons.

| Source | Provides | Best use | Caveat | Links |
|---|---|---|---|---|
| Microsoft Planetary Computer | STAC-indexed Sentinel-2 L2A, Sentinel-1 SAR, and context layers. | Custom `T x H x W x C` chips around CropHarvest points, EuroCrops parcels, or sampled croplands. | Service status/access should be verified before large dependency. | [catalog](https://planetarycomputer.microsoft.com/catalog), [STAC docs](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/) |
| AWS Sentinel-2 L2A COGs / Earth Search | Sentinel-2 L2A scenes as cloud-optimized GeoTIFFs with STAC metadata. | Efficient S2 chip extraction and cloud-native preprocessing. | Optical only. | [AWS registry](https://registry.opendata.aws/sentinel-2-l2a-cogs/) |
| OPERA Sentinel-1 RTC | Radiometric terrain corrected Sentinel-1 backscatter. | Better SAR channel than hand-rolled GRD preprocessing. | Coverage/product details need checking for target years/regions. | [AWS registry](https://registry.opendata.aws/nasa-operal2rtc-s1v1/), [docs](https://sentinel-s1-rtc-indigo-docs.s3-us-west-2.amazonaws.com/index.html) |
| Copernicus Data Space | Direct Copernicus Sentinel access. | Fallback/full-fidelity source. | More operational friction than cloud-optimized STAC/COG. | [portal](https://dataspace.copernicus.eu/), [S1 docs](https://documentation.dataspace.copernicus.eu/Data/Sentinel1.html) |
| Google Earth Engine | Sentinel-2 SR Harmonized, Sentinel-1 GRD, ERA5/ERA5-Land, MODIS, SMAP, SRTM, land cover. | Rapid geospatial joins and covariate extraction. | Operational/export quotas and reproducibility need care. | [S2 SR Harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED), [S1 GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) |
| NASA HLS | Harmonized Landsat/Sentinel-2 surface reflectance. | Longer temporal coverage and denser optical revisit at 30m. | Loses 10m detail and Sentinel-2 red-edge specificity requires care. | [products](https://hls.gsfc.nasa.gov/data-products/), [access](https://hls.gsfc.nasa.gov/data-access-and-tools/) |
| Copernicus legal notice | Sentinel data usage terms. | Licensing/usage reference. | Not a data source. | [legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) |

## Climate, Soil, Terrain, And Water

| Priority | Source | Provides | Best use | Caveat | Links |
|---:|---|---|---|---|---|
| 0 | AgERA5 | Daily agriculture-oriented climate variables. | Main climate source for global crop dynamics. | Coarse grid; cache point/patch summaries. | [CDS](https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators) |
| 0 | ERA5-Land | Hourly land climate at about 9km. | Backup or complement to AgERA5. | Requires daily aggregation. | [ECMWF](https://www.ecmwf.int/en/era5-land) |
| 1 | Daymet | Daily 1km weather over North America. | Higher-resolution U.S./North America yield and stress work. | Not global. | [Daymet](https://daymet.ornl.gov/), [ORNL guide](https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4R1.html) |
| 1 | CHIRPS | Global-ish daily precipitation at 0.05 degrees from 1981 onward. | Rainfall anomaly and drought features. | Precipitation only. | [NOAA ERDDAP](https://coastwatch.pfeg.noaa.gov/erddap/griddap/chirps20GlobalDailyP05.html) |
| 1 | SRTM | Global 30m elevation. | Elevation, slope, aspect, terrain position. | Static. | [GEE](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) |
| 1 | SoilGrids | Global soil properties at 250m and multiple depths. | Soil texture, pH, organic carbon, cation exchange capacity, coarse fragments. | Coarse and modeled. | [ISRIC](https://www.isric.org/explore/soilgrids) |
| 2 | HWSD v2.0 | Global soil inventory at about 1km. | Backup/coarser soil covariates. | Coarser than SoilGrids. | [FAO](https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v20/en/) |
| 2 | SMAP L4 | 3-hourly global surface/root-zone soil moisture at about 9km. | Water-stress context and drought-aware features. | Coarse; may duplicate reanalysis signal. | [GEE](https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_008), [NASA GMAO](https://gmao.gsfc.nasa.gov/gmao-products/smap-l4/) |
| 2 | MODIS vegetation products | NDVI/EVI and LAI/FPAR time series. | Coarse phenology/yield context. | Too coarse for main Sentinel input. | [MOD13Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1), [MOD15A2H](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD15A2H) |

## Yield, Stress, And Operational Agriculture

Do not use these until representation transfer and at least one external crop-type benchmark are under control. They are high-value but easy to misuse because labels are noisy, aggregated, management-dependent, or from a different sensing regime.

| Priority | Source | What it is | Best use | Limitation | Links |
|---:|---|---|---|---|---|
| 1 | YieldSAT | Field/subfield crop yield benchmark with Sentinel-2, weather, soil, and topography. | High-resolution yield-related evaluation. | New dataset; availability/workflow must be verified. | [site](https://yieldsat.github.io/), [paper](https://arxiv.org/abs/2604.00940) |
| 1 | CropNet | U.S. county-level crop yield dataset with Sentinel-2, WRF-HRRR weather, and USDA crop data. | Climate-aware yield modeling. | County labels are coarse relative to field/pixel embeddings. | [Hugging Face](https://huggingface.co/datasets/CropNet/CropNet), [paper](https://arxiv.org/abs/2406.06081) |
| 2 | SustainBench crop yield | Sustainability benchmark with crop yield task. | Compatibility with sustainability ML work. | County/region aggregation and older baselines. | [SustainBench](https://sustainlab-group.github.io/sustainbench/) |
| 2 | Agriculture-Vision | Aerial RGB/NIR farmland stress/anomaly segmentation. | Stress/anomaly concept benchmark. | 10cm aerial imagery, not Sentinel-scale satellite imagery. | [paper](https://arxiv.org/abs/2001.01306), [site](https://agriculture-vision.intelinair.com/) |
| 2 | USDA QuickStats + CDL | U.S. county yields and annual crop maps. | U.S. yield/crop-type sanity checks and derived labels. | U.S.-specific; CDL is a map product, not clean independent ground truth everywhere. | [QuickStats API](https://quickstats.nass.usda.gov/api), [metadata](https://catalog.data.gov/dataset/quick-stats-agricultural-database-api), [CDL context](https://data.nass.usda.gov/Education_and_Outreach/Reports%2C_Presentations_and_Conferences/reports/conferences/JSM-2022/JSM2022_Murphy_final.pdf) |

## Field Boundaries And Object-Level Context

| Priority | Source | What it is | Best use | Limitation | Links |
|---:|---|---|---|---|---|
| 1 | Fields of The World | Global agricultural field boundary segmentation benchmark. | Field-level pooling, boundary-aware masking, dense segmentation evaluation. | Benchmark first; product coverage/access must be checked separately. | [site](https://fieldsofthe.world/index.html), [paper](https://arxiv.org/abs/2409.16252) |
| 1 | Global 10m field boundary map | Global field boundary map with billions of polygons. | Potential field object prior if accessible. | Very new; license/access/quality review required. | [paper](https://arxiv.org/abs/2605.11055) |
| 2 | CropSight-US | Object-based U.S. crop type ground truth using street-view and Sentinel-2. | U.S. object-level crop-type evaluation and label validation. | U.S.-only; integration work needed. | [ESSD](https://essd.copernicus.org/articles/18/3069/2026/) |

## Generic EO Benchmarks

These can help compatibility with broader geospatial foundation model reporting, but they should not define the core agricultural claim.

| Source | Use | Why not first | Links |
|---|---|---|---|
| BigEarthNet-MM | Large Sentinel-1/Sentinel-2 land-cover benchmark. | Not crop-specific and not a crop dynamics benchmark. | [HF mirror](https://huggingface.co/datasets/lc-col/bigearthnet), [GFM-Bench mirror](https://huggingface.co/datasets/GFM-Bench/BigEarthNet) |
| SEN12MS | Sentinel-1/Sentinel-2/MODIS land-cover data. | Generic land cover; older task framing. | [paper](https://arxiv.org/abs/1906.07789) |
| EuroSAT | Easy sanity benchmark for image encoders. | Single-date scene classification; too small/simple for the claim. | [GitHub](https://github.com/phelber/eurosat) |
| CloudSEN12 | Cloud/cloud-shadow benchmark with Sentinel-1/2. | Useful only if cloud masking becomes a model component; not crop-specific. | [site](https://cloudsen12.github.io/Legacy/index.html) |
| PANGAEA | Broad GFM benchmark protocol. | Protocol reference, not a crop-specific data source. | [paper](https://arxiv.org/abs/2412.04204) |

## Evaluation And Protocol Papers

- Generalizability of foundation models for crop type mapping: https://arxiv.org/abs/2409.09451
  - Evaluates SSL4EO-S12, SatlasPretrain, and ImageNet-style weights on five crop classification datasets across five continents. Useful for external crop-transfer framing.
- Deploying geospatial foundation models in WorldCereal: https://arxiv.org/abs/2508.00858
  - Operational crop-mapping case study using Presto inside WorldCereal. Useful for deployment constraints and domain adaptation framing.
- Harvesting AlphaEarth: https://arxiv.org/abs/2601.00857
  - Evaluates AlphaEarth Foundation embeddings on U.S. agricultural tasks. Flags spatial transferability, interpretability, and time sensitivity issues.
- PANGAEA: https://arxiv.org/abs/2412.04204
  - Standardized evaluation protocol for GFMs across datasets, tasks, resolutions, sensors, and temporalities.
- No One Knows the State of the Art in Geospatial Foundation Models: https://arxiv.org/abs/2605.12678
  - Audit arguing GFM comparisons are unreliable unless evaluations, pretraining controls, weights, and protocols are consistent.

## Canonical Data Objects

### 1. CropHarvest v2 point tensor

Purpose: fixed strict-transfer evaluation input.

Shape:

```text
N x 12 x C x 1 x 1
```

Required fields:

- S2: `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`, `NDVI`;
- S1: `VV`, `VH`;
- context: `temperature`, `precipitation`, `elevation`;
- source dataset id, country, region/group id, label type, and heldout split;
- modality/time masks;
- exact band-name metadata in the Zarr store.

This does not make CropHarvest enough for foundation pretraining. It makes evaluation less lossy and keeps `[7]` honest.

### 2. SSL4EO-S12 v1.1 patch corpus

Purpose: first real patch-time-series pretraining run.

Initial target:

```text
N x 4 x 16 x 16 x C
```

Start with:

- S2 L2A bands matching the `[7]` S2 contract;
- S1 channels matching `VV`, `VH`;
- NDVI;
- DEM/elevation;
- cloud mask;
- lat/lon and timestamp metadata.

For `[7]`, context should be exactly:

```text
temperature, precipitation, elevation
```

Crop-prior channels should be used only in a later ablation once the same external prior exists for both pretraining and evaluation.

### 3. Agriculture-filtered SSL4EO subset

Purpose: avoid training mostly generic EO when the project goal is crop monitoring.

Sampling plan:

- positive agriculture samples: high WorldCereal/WorldCover/Dynamic World cropland confidence;
- hard near-agriculture negatives: grassland, shrubland, bare, wetland, forest near cropland;
- geography-stratified sampling: preserve region/country/biome/source metadata;
- cloud-stress strata: explicitly include cloudy/partially missing optical sequences.

This is more important than simply downloading more data.

### 4. External crop benchmark bundle

Prepare one first:

1. EuroCropsML if the easiest path is few-shot parcel crop type.
2. ZueriCrop if spatial patch evidence is more important.
3. TimeSen2Crop if large pixel-time-series crop-type transfer is fastest.
4. PASTIS if dense spatial segmentation becomes part of the claim.

Do not prepare all of them before the patch pretraining pipeline works.

### 5. Raw Sentinel chip builder

Purpose: graduate from four seasonal timestamps to real irregular crop seasons.

Target:

```text
N x 12-36 x H x W x C
```

with:

- actual observation dates;
- `H = W = 16` or `32` first, larger later;
- S2, S1, cloud/quality masks, climate summaries, terrain, crop priors where valid, and availability masks.

This is the stronger long-term data object.

## CropHarvest Download URLs

- Labels GeoJSON: https://zenodo.org/records/5037916/files/labels.geojson
- Features tarball: https://zenodo.org/records/5037916/files/features.tar.gz
- Labels direct download: https://zenodo.org/records/5037916/files/labels.geojson?download=1
- Features direct download: https://zenodo.org/records/5037916/files/features.tar.gz?download=1

## Data Shape Reminder

- Pixel time series: `T x C`, where `T` is observation time and `C` contains S2, S1, climate, terrain, and timing channels.
- Patch time series: `T x H x W x C`; necessary for spatial context or segmentation claims.
- Sentinel-2: optical/spectral bands including visible, red-edge, NIR, and SWIR. Cloud contamination is the main missingness issue.
- Sentinel-1: radar backscatter, usually VV/VH. It is cloud-robust and captures structure, roughness, moisture, and canopy effects.
- ERA5/AgERA5: coarser climate variables aligned to field/pixel over time.
- Timing: day-of-year, phenological windows, or growing-degree-day features should be model inputs, not downstream-only metadata.

## What Not To Do

- Do not spend serious compute scaling the old `1 x 1` CropHarvest-only setup unless the experiment isolates a specific architectural hypothesis.
- Do not claim stress or yield from crop/non-crop labels.
- Do not treat CDL, WorldCover, Dynamic World, or WorldCereal as clean crop-type ground truth. They are priors, masks, pseudo-labels, or sampling tools unless the experiment is explicitly designed around them.
- Do not mix target regions into SSL pretraining if the claim is strict target-domain exclusion.
- Do not compare against broad EO models without stating whether numbers are copied from papers, rerun on our splits, or evaluated with different input availability.

## Data To Prepare Next

1. `ssl4eo_s12_v11_48k.zarr`
   - Use `49,152` SSL4EO samples for the first bounded `[7]` screen.
   - Keep the stream-and-evict source cache capped at `60 GiB`.
   - Treat this as a packaged SSL4EO architecture screen, not a true agriculture-masked corpus.
   - Prefer WebDataset/Zarr over raw GeoTIFF tarballs.
   - Verify S2 L2A, S1, NDVI, DEM/elevation, timestamps, coordinates, and cloud masks.
   - Match the `[7]` channel contract exactly.

2. `cropharvest_v2.zarr`
   - Include NDVI and elevation.
   - Preserve raw spectral/radar stats.
   - Preserve all split/group/source metadata needed for strict heldout tests.
   - Match the `[7]` channel contract exactly.

3. Climate join cache
   - AgERA5 first; ERA5-Land fallback.
   - Cache daily or seasonal summaries by sample id/date/location.
   - Include temperature, precipitation, radiation/vapor pressure later if the channel contract is extended deliberately.

4. Agriculture sampling masks
   - WorldCereal 2021, ESA WorldCover 2021, and Dynamic World are enough to start.
   - For `[7]`, use them for sampling/filtering, not as a model channel.

5. One external crop-type benchmark
   - First choice: EuroCropsML.
   - Second choice: ZueriCrop.
   - Third choice: TimeSen2Crop.

6. Optional later: YieldSAT or CropNet
   - YieldSAT is more aligned with field/subfield yield.
   - CropNet is easier for U.S. county-level yield but less aligned with pixel/field embeddings.

## Current Data Decision

The next serious data target is:

```text
SSL4EO-S12 v1.1 + AgERA5/ERA5-Land + agriculture-filtered sampling -> Patch-JEPA pretraining
CropHarvest v2 + EuroCropsML/ZueriCrop/TimeSen2Crop -> evaluation
```
