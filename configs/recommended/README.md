Recommended settings per dashboard dataset = the settings that dataset's final run actually used
(`uav_nidd.json` = v14m-NIDD-CPU-FINAL, `isot.json` = v14m ISOT with FedWGCA quartile 0.15 on CPU,
`isot72.json` = ISOT72-plain-FINAL, `nidd_plain.json` = NIDD-plain-FINAL), plus `compute.workers`
(how many jobs run at once, chosen for 24 GB RAM). The dashboard's training form starts from these files,
never writes to them, and resets to them when locked or after a restart.
