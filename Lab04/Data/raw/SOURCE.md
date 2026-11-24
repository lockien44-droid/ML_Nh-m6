# California Housing

- Loader: sklearn.datasets.fetch_california_housing (as_frame=True).
- Documentation: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- Download archive used by scikit-learn: https://ndownloader.figshare.com/files/5976036
- Dataset background: 1990 California census, 20,640 census block groups before pipeline duplicate filtering, eight predictors.
- Target: median house value in units of 100,000 USD. Source target is capped near 5.00001; not individual-house sale prices or current valuations.
- Downloaded CSV is a local export from the loader; source checksum for the actual CSV is recorded in every run.json. Consult the upstream source for usage terms; no new license is asserted here.
- Data/raw/california_housing.csv and Data/cache/ are generated/downloaded, ignored by Git.
