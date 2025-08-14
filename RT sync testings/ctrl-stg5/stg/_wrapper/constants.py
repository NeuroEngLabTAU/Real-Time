CURRENT_RESOLUTION_in_nA = "current_resolution_in_nA"
CURRENT_RANGE_in_uA = "current_range_in_uA"
TIME_RESOLUTION_in_us = "time_resolution_in_us"
OUTPUT_RATE_in_Hz = "output_rate_in_Hz"

STG5_specification = {
    CURRENT_RESOLUTION_in_nA: [6.25, 62.5, 625],
    CURRENT_RANGE_in_uA: [160, 1.6e3, 16e6],
    TIME_RESOLUTION_in_us: 5,
}
STG5_specification[OUTPUT_RATE_in_Hz] = 1e6 / STG5_specification[TIME_RESOLUTION_in_us]

STG4002_specification = {
    CURRENT_RESOLUTION_in_nA: 20, #In the specification sheet writeen as 2000nA, but in the MC_Simulus is 20nA
    CURRENT_RANGE_in_uA: 160,
    TIME_RESOLUTION_in_us: 20,
}
STG4002_specification[OUTPUT_RATE_in_Hz] = 1e6 / STG4002_specification[TIME_RESOLUTION_in_us]