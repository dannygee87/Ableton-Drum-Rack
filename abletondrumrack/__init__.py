# -*- pretty cool init mate? -*-

import sys
from abletondrumrack.ableton_drum_rack import DrumRack
from abletondrumrack.tools import create_sample_database, update_sample_details, add_new_samples, query, get_main_freq, check_db_for_dups, remove_samples_from_db, create_drumrack_from_df


__all__ = [
    'DrumRack',
    'create_sample_database',
    'update_sample_details',
    'remove_samples_from_db',
    'add_new_samples',
    'query',
    'get_main_freq',
    'check_db_for_dups',
    'create_drumrack_from_df'
    ]