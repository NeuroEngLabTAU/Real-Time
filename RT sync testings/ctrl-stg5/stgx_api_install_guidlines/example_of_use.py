from stg.api import PulseFile, STG4000, STG5

# stg = STG4000()
stg = STG5()
stg.download(0, *PulseFile().compile())
stg.start_stimulation([0])