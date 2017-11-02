import os
base = "/".join(os.path.realpath(__file__).split("/")[:-1])+ "/tools"

manticore_policies = ["random", "uncovered", "branchlimited"]
manticore_dist_symb_bytes = ["sparse", "dense"]

grrshot_path = base + "/grr/bin/grrshot"
grrplay_path = base + "/grr/bin/grrplay"

grr_mutators = [
            "splice",           # Splice one receive somewhere into the testcase.
            "splice_chunked",   # Chunke then splice receives around.
            "dropper",          # Delete some of the receives.
            "random",           # Replace the data in some receives with random data.
            "bitflip1",         # Flip bit 1 (in some of the receives).
            "bitflip2",         # Flip bit 2.
            "bitflip3",         # Flip bit 3.
            "bitflip4",         # Flip bit 4.
            "bitflip5",         # Flip bit 5.
            "bitflip6",         # Flip bit 6.
            "bitflip7",         # Flip bit 7.
            "bitflip8",         # Flip bit 8.
            "bitflip2_2",       # Flip bits 1 and 2.
            "bitflip3_2",       # Flip bits 2 and 3.
            "bitflip4_2",       # Flip bits 3 and 4.
            "bitflip5_2",       # Flip bits 4 and 5.
            "bitflip6_2",       # Flip bits 5 and 6.
            "bitflip7_2",       # Flip bits 6 and 7.
            "bitflip8_2",       # Flip bits 7 and 8.
            "bitflip4_4",       # Flip bits 1, 2, 3, and 4.
            "bitflip6_4",       # Flip bits 3,4, 5, and 6.
            "bitflip8_4",       # Flip bits 5, 6, 7, and 8.
            "bitflip8_8",       # Flip all bits.
            "inf_bitflip_random",
            "random",
            "inf_radamsa_chunked",
            "inf_radamsa_spliced",
            "inf_radamsa_concat",
            "inf_chunked_repeat",
]


afl_path = base + "/afl-cgc/bin"
aflfuzz_path = base + "/afl-cgc/bin/afl-fuzz"
afltmin_path = base + "/afl-cgc/bin/afl-tmin"
aflcmin_path = base + "/afl-cgc/bin/afl-cmin"
aflshowmap_path = base + "/afl-cgc/bin/afl-showmap"


os.environ["AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES"] = "1"
os.environ["AFL_SKIP_CPUFREQ"] = "1"
