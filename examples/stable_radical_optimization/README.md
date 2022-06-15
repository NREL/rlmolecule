## Stable Radical Optimization

To run on eagle:
bash run_eagle.sh <run_id> config/config_eagle.yaml



### notes for developing on code ocean

* Could get a shared memory database across processes going with tmpfs?
https://stackoverflow.com/questions/42884087/how-and-when-to-use-dev-shm-for-efficiency/42884337#42884337
`mount -t tmpfs tmpfs /mnt/tmp`

