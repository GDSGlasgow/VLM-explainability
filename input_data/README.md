# Local input data

Only `cityscapes_labels.csv` is versioned because it is a small, public semantic
label mapping required by the masking pipeline.

The following study-specific inputs are intentionally ignored by Git and must be
created locally when needed:

- `sample_coordinates.csv`: columns `latitude,longitude,city`
- `sequence_keys.csv`: columns `index,sequence_id,city,notes`

Do not place API tokens or other credentials in these files. Set the Mapillary
token through the `MAPILLARY_TOKEN` environment variable.
