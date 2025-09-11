""" This file handles Hashing, SQL, Frames, Duration. """

import os
import hashlib
import sqlite3
from tokenize import String
import numpy as np
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import PureWindowsPath, PurePosixPath, PurePath
from scipy import fft
from scipy.io import wavfile




#import matplotlib.pyplot as plt
#from IPython.display import Audio

### helper functions
def changePathToPosix(orgPath):
    if os.name == 'nt':
        return orgPath
    else:
        return PureWindowsPath(orgPath).as_posix()


def frequency_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    #x = fft(x) / n  # fft computing and normalization
    x = fft.fft(x) / n # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)


# https://www.ibm.com/docs/en/zos/2.3.0?topic=attributes-block-size-record-length
BLOCK_SIZE = 65536
# Default db path
DB_PATH = os.path.expanduser("~") + '{}ableton_samples.db'.format(os.sep)


def sample_hash(path):
    # Currently not used
    """
        Create md5 hash for sample file.
    """
    purepath = changePathToPosix(path)
    file_hash = hashlib.md5() 
    with open(purepath, 'rb') as file:
        fileblock = file.read(BLOCK_SIZE)
        while len(fileblock) > 0:
            file_hash.update(fileblock)
            fileblock = file.read(BLOCK_SIZE)
    hash = file_hash.hexdigest()
    return hash
    

def sample_frames_length(path, loginfo):
    """
        Get frames and duration of sample.
        If logging true store data in database.
    """
    # If extraction fails set sample to not supported.
    try:
        purepath = changePathToPosix(path)
        f = sf.SoundFile(purepath)
        duration = f.frames / f.samplerate
        if loginfo:
            update_frames_length(f.frames, duration, path)
            
        return f.frames, duration
        
    except Exception as e:
        connection = sqlite3.connect(DB_PATH)
        cur = connection.cursor()
        cur.execute('UPDATE SAMPLE_PATHS SET SUPPORTED = 0 WHERE FULL_FILE_PATH = "{}"'.format(path))
        connection.commit()
        connection.close() 


def sample_main_freq(path, loginfo):
        purepath = changePathToPosix(path)
        #print(possample)
        err = False
        try:
            sr, signal = wavfile.read(purepath) 
        except:
            err = True
            return err
        
        
        if signal.ndim == 1:
            y=signal
        else:
            y = signal[:, 0]  # use the first channel (or take their average, alternatively)

        t = np.arange(len(y)) / float(sr)

        frq, X = frequency_spectrum(y, sr)

        indexofmaxval = np.argmax(X, axis=-1) # get the index from the biggest value in array
        mainfreq = frq[indexofmaxval] # value of item with index from above = Frequency?
        mainfreq = mainfreq.round(2)    
        update_main_frequency(mainfreq, path)


def sample_size(path, loginfo):
    """
        Get sample file size.
    """
    purepath = changePathToPosix(path)
    sample_size = os.path.getsize(purepath)     
    if loginfo:
        update_sample_size(sample_size, path)
    else:
        return sample_size


def create_sample_df(path):
    """
        Get all sample paths inside the given folder.
        Only looks for wav files.
    """
    # Get full paths for samples.
    dirs = [folder[0] for folder in os.walk(path)]
    # Unacceptable.
    dirs = [x for x in dirs if len([x for x in os.listdir(x) if '.wav' in x])]

    folder_path, sample_name, ff_path, folder_name, hash = [], [], [], [], []

    for _path in dirs:
        for _sample in os.listdir(_path):
            if _sample.endswith('.wav') and _sample[0] != '.':
                full_sample_path = _path + '\\' + _sample
                ff_path.append(full_sample_path)
                folder_path.append(_path)
                folder_name.append(_path.split('\\')[-1])
                sample_name.append(_sample)
                hash.append(sample_hash(full_sample_path))
    # Create a dataframe                 
    dirs_add = pd.DataFrame(
        {'FOLDER_PATH': folder_path,
         'FOLDER_NAME': folder_name,
         'SAMPLE_NAME': sample_name, 
         'FULL_FILE_PATH': ff_path,
         'SAMPLE_HASH': hash,
         }
    )
    # Add empty columns for updates later on. Currently time consuming on build if many samples provided.
    dirs_add['SAMPLE_SIZE'] = np.nan
    dirs_add['FRAMES'] = np.nan
    dirs_add['LENGTH'] = np.nan
    dirs_add['MAINFREQ'] = np.nan
    dirs_add['SUPPORTED'] = np.nan
    
    return dirs_add
    

#######################
## Sqlite3 functions ##
#######################

def query(_query):
    """
        Takes query inputs from user.
    """
    try:
        connection = sqlite3.connect(DB_PATH)
        result = pd.read_sql(_query, connection)
        connection.close()
        
        if result.empty:
            raise ValueError('No samples found for query')
            
        return result
        
    except Exception as e:
        if 'no such table' in str(e):
            print('Database or Table not Found')
        else:
            print(e)


def update_frames_length(f, d, p):
    """
        Update sample in database for frames and duration.
    """
    connection = sqlite3.connect(DB_PATH)
    cur = connection.cursor()
    cur.execute('UPDATE SAMPLE_PATHS SET FRAMES = {}, LENGTH = {} WHERE FULL_FILE_PATH = "{}"'.format(f, d, p))
    connection.commit()
    connection.close()    

def update_main_frequency(freq, p):
    """
        Update sample in database for frames and duration.
    """
    connection = sqlite3.connect(DB_PATH)
    cur = connection.cursor()
    cur.execute('UPDATE SAMPLE_PATHS SET MAINFREQ = {} WHERE FULL_FILE_PATH = "{}"'.format(freq, p))
    connection.commit()
    connection.close()        

def update_sample_size(size, p):
    """
        Update sample in database for frames and duration.
    """
    connection = sqlite3.connect(DB_PATH)
    cur = connection.cursor()
    cur.execute('UPDATE SAMPLE_PATHS SET SAMPLE_SIZE = {} WHERE FULL_FILE_PATH = "{}"'.format(size, p))
    connection.commit()
    connection.close()        

def update_sample_hash(hash, p):
    """
        Update sample in database for frames and duration.
    """
    
    connection = sqlite3.connect(DB_PATH)
    cur = connection.cursor()
    cur.execute('UPDATE SAMPLE_PATHS SET SAMPLE_HASH = {} WHERE FULL_FILE_PATH = "{}"'.format(hash, p))
    connection.commit()
    connection.close()        

def update_sample_IN_DRUMRACK(drumrackname, p):
    """
        Update sample in database with assigned drumrack name
    """
    
    connection = sqlite3.connect(DB_PATH)
    cur = connection.cursor()
    cur.execute('UPDATE SAMPLE_PATHS SET IN_DRUM_RACK = "{}" WHERE FULL_FILE_PATH = "{}"'.format(drumrackname, p))
    connection.commit()
    connection.close()        
    

def delete_samples_from_database(path):
    connection = sqlite3.connect(DB_PATH)
    cur = connection.cursor()
#    sql_delete_query = """DELETE from SAMPLE_PATHS WHERE FULL_FILE_PATH = '{}' """
    cur.execute('DELETE from SAMPLE_PATHS WHERE FULL_FILE_PATH = "{}"'.format(path))
#   cur.execute('UPDATE SAMPLE_PATHS SET FRAMES = {}, LENGTH = {} WHERE FULL_FILE_PATH = "{}"'.format(f, d, p))
    connection.commit()
    connection.close()    


def create_sample_database(path):
    """
        Creates a database for the samples in a given path.
        All subdirectories in path are included.
    """
    if 'ableton_samples.db' in os.listdir(os.path.expanduser("~")):
        # Make sure user wants to recreate if database exists. It's time consuming.
        _input = input('Database already exists. Do you want to overwrite? y/n:')
        if _input == 'y':
            samples_df = create_sample_df(path)
            connection = sqlite3.connect(os.path.expanduser("~") + '{}ableton_samples.db'.format(os.sep))
            samples_df.to_sql('SAMPLE_PATHS', connection, if_exists='replace', index=False)
            connection.close()
            
            return 'Database Succesfully Created'
    else:
        samples_df = create_sample_df(path)
        connection = sqlite3.connect(os.path.expanduser("~") + '{}ableton_samples.db'.format(os.sep))
        samples_df.to_sql('SAMPLE_PATHS', connection, if_exists='replace', index=False)
        connection.close()
        
        return 'Database Succesfully Created'
        
def remove_samples_from_db(samples_df):
    for i ,r in samples_df.iterrows():
        delete_samples_from_database(r['FULL_FILE_PATH'])
       

def update_sample_details(samples_df, force=False):
    """
        Update frames and duration for samples given a samples_dataframe.
        !!! Warning a full database update may take some time depending on the amount of samples.
    """
    sample_types = samples_df
    for i ,r in sample_types.iterrows():
        if force:
            sample_frames_length(r['FULL_FILE_PATH'], loginfo=True)
            sample_size(r['FULL_FILE_PATH'], loginfo=True)   
        else:
            if r['FRAMES'] is None:
                sample_frames_length(r['FULL_FILE_PATH'], loginfo=True)
                sample_size(r['FULL_FILE_PATH'], loginfo=True)   
            else:
                continue
        
#     if path:
#         z = "select * from SAMPLE_PATHS where FULL_FILE_PATH LIKE '%{}%' AND LENGTH IS NULL".format(path)
#         #sample_types = query(z)
#         sample_types = samples_array
#         for i ,r in sample_types.iterrows():
#             sample_frames_length(r['FULL_FILE_PATH'], loginfo=True)
# #            sample_main_freq(r['FULL_FILE_PATH'], loginfo=True)   
#             sample_size(r['FULL_FILE_PATH'], loginfo=True)   
#     else:
#         z = "select * from SAMPLE_PATHS"
#         sample_types = query(z)
#         for i ,r in sample_types.iterrows():
#             sample_frames_length(r['FULL_FILE_PATH'], loginfo=True)
# #            sample_main_freq(r['FULL_FILE_PATH'], loginfo=True)        
#             sample_size(r['FULL_FILE_PATH'], loginfo=True)   

def get_main_freq(samples_df):
    """
        Analyse sample main freq of given samples-array and write it to DB.
    """
    #z = "select * from SAMPLE_PATHS where FULL_FILE_PATH LIKE '%{}%' AND LENGTH IS NULL".format(path)
    #z = 'SELECT * FROM SAMPLE_PATHS WHERE LENGTH < 1 AND MAINFREQ IS NULL AND FOLDER_NAME LIKE "%PERC%"'
    
   # print(len(sample_types))
    #return sample_types
    error = False
    errsamples = []
    for i ,r in samples_df.iterrows():
        if r['MAINFREQ'] is None and r['LENGTH'] < 5:   # check if MAINFREQ is NULL and Sample_Length < 5 sek
            error = sample_main_freq(r['FULL_FILE_PATH'], loginfo=True)
            if error == True:
                errsamples.append(r['FULL_FILE_PATH'])
        else:
            
            continue
    return errsamples

def add_new_samples(path):
    """
        Inserts samples from a given a path. Returns df with not imported samples because they are already in DB.
    """
    new_samples_df = create_sample_df(path)
    samples_already_inDB = query("SELECT * FROM SAMPLE_PATHS")
    
    ### Check if newsamples already in DB on SAMPLE_HASH
    cond = new_samples_df["SAMPLE_HASH"].isin(samples_already_inDB["SAMPLE_HASH"])
    duplicatesamples_indices_list = [i for i in cond.index if cond[i]] # erstelle neue Liste mit allen True Indiezes
    notimported_becauseDups_df = new_samples_df[new_samples_df.index.isin(duplicatesamples_indices_list)] 
       
    new_samples_df.drop(new_samples_df[cond].index, inplace=True)
    if new_samples_df.empty:
        return("samples_to_import_df is empty... all samples are already in DB")
    else:
        connection = sqlite3.connect(os.path.expanduser("~") + '{}ableton_samples.db'.format(os.sep))
        new_samples_df.to_sql('SAMPLE_PATHS', connection, if_exists='append', index=False)
        connection.close()
        return notimported_becauseDups_df


def check_db_for_dups():
    z = 'SELECT * FROM SAMPLE_PATHS'
    allsamples = query(z)
    erg = allsamples['SAMPLE_HASH']
    
    return erg


def create_drumrack_from_df(rackname_base, rackname_number_to_begin_with, samples_df, nr_samples_per_rack=48 ):
    """
        creating dumracks with nr_samples_per_rack samples and name it "rackname_base"_"rack_number_to_begin_with" ++ 
    """
    from abletondrumrack import DrumRack
    rack = DrumRack()
    rack.save_path = r'C:\Users\danie\Documents\1-programming\Python\venvs\abletondrumrack\presets-out'


    # check if some samples are already part of a drumrack. if yes abort
    samples_allready_in_rack = samples_df[samples_df[["IN_DRUM_RACK"]].notnull().all(1)]
    if samples_allready_in_rack.shape[0] == 0:
        numberfullracks = len(samples_df) // nr_samples_per_rack
        #lastrack = len(samples_df) % nr_samples_per_rack

        for i in range(numberfullracks):
            racknr = i + rackname_number_to_begin_with
            rackname = rackname_base + "_" + str(racknr)
            samplesstart = i * nr_samples_per_rack
            samplesend = (i + 1) * nr_samples_per_rack 
            #print(samplesstart)
            #print(samplesend)
            samplesforRack = samples_df[samplesstart:samplesend]
            rack.make_moi_drum_rack(samples=samplesforRack, fname=rackname, slots=nr_samples_per_rack)

        lastrack_sample_start = numberfullracks * nr_samples_per_rack
        lastracknr = racknr + 1
        lastrackname = rackname_base + "_" + str(lastracknr)

        samplesforRack = samples_df[lastrack_sample_start:]
        rack.make_moi_drum_rack(samples=samplesforRack, fname=lastrackname, slots=nr_samples_per_rack)

    else:
        return "Some samples of samples_df are already in a Rack. Exit"



