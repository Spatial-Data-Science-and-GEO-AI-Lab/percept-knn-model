#!/usr/bin/env python3
#
# clip_retrieval_knn.py
#
# Runs a K-nearest neighbour analysis on CLIP-encoded vectors. This will also
# generate said encoded vectors from a directory of images, using the
# 'clip-retrieval' tool (available separately), and it will also generate text
# prompts and corresponding encoded vectors if the environmental features
# option is enabled.
#
# There are many options and the program is designed to work with the survey
# data from the Percept project:
#   https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/percept
# More details can be found in the README.md file.
#
# Copyright (2024): Matthew Danish
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
import argparse
import regex
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.extmath import softmax
from scipy.spatial import geometric_slerp
import json
import csv
import sys
import subprocess
from pathlib import Path
from transformers import AutoProcessor, CLIPVisionModel
from tqdm import tqdm
from PIL import Image
import pyarrow.parquet as pq
import geopandas as gpd
import pandas as pd
from random import shuffle, seed

parser = argparse.ArgumentParser(prog='clip_retrieval_knn', description='K-nearest neighbour on CLIP encoded vectors')
parser.add_argument('--images-dir', '-i', metavar='DIRECTORY', type=str, help='Directory with images to be processed with clip-retrieval tool', default=None)
parser.add_argument('--embeddings-dir', '-e', metavar='DIRECTORY', type=str, help='Directory for embeddings output of clip-retrieval tool', default=None)
parser.add_argument('--clip-model', '-M', metavar='MODELNAME', type=str, help='CLIP model name', required=True)
parser.add_argument('--other-clip-retrieval-args', metavar='ARGS', type=str, help='Other command line args to pass to clip-retrieval', default='')
parser.add_argument('--geojson', '-g', metavar='FILENAME', type=str, help='File with GeoJSON data from survey', required=True)
parser.add_argument('--demographics', '-d', metavar='FILENAME', type=str, help='CSV File with demographic data per rating from survey', default=None)
parser.add_argument('-k', metavar='K', help='Value of K (number of nearest neighbours to include in cluster) or comma-separated list of k-values to try.', default=10)
parser.add_argument('--training-split', metavar='FLOAT', help='Portion of data to use for \'training\', value between 0 and 1 (default: 0.8)', default=0.8, type=float)
parser.add_argument('--randomize', action='store_true', help='Randomly shuffle the data before splitting into training and testing sets.', default=False)
parser.add_argument('--random-seed', metavar='INT', help='Seed for random number generator.', default=None, type=int)
parser.add_argument('--stratified', action='store_true', help='Use stratified sampling (stratified by rating).', default=False)
parser.add_argument('--environmental', action='store_true', help='Add environmental features into the model', default=False)
parser.add_argument('--environmental-method', metavar='METHOD', type=str, help='One of: append, average, slerp', default='append')
parser.add_argument('--environmental-text-dir', metavar='DIR', type=str, help='Path to dir containing prompt files for environmental vars', default=None)
parser.add_argument('--prompt-style', metavar='NUM', type=int, help='One of: 0, 1', default=0)
parser.add_argument('--results-log', '-L', metavar='FILENAME', type=str, help='Append the results to this file (CSV format)', default=None)
parser.add_argument('--normalization-method', metavar='METHOD', type=str, help='softmax10** (default), softmax or divbysum', default='softmax10**')
parser.add_argument('--skip-cache', action='store_true', help='Do not look for or read any cached data.', default=False)
parser.add_argument('--read-only', action='store_true', help='Do not write any data to disk (cache or otherwise).', default=False)
parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output to minimum.', default=False)
parser.add_argument('--extra-assertions', action='store_true', help='Run additional assertions for testing purposes.', default=False)
parser.add_argument('--gender', metavar='GENDER,...', help='Comma-separated list of surveyed people\'s genders to include in analysis', default=None)
parser.add_argument('--region', metavar='REGION', help='Include in analysis only those ratings from people who claim to be from this stated region (NL, non-NL)', default=None)
parser.add_argument('--age', metavar='AGE_MIN,AGE_MAX', help='Include in analysis only those ratings from people who claim to be from this stated age range', default=None)
parser.add_argument('--education', metavar='LEVEL,...', help='Comma-separated list of surveyed people\'s education level to include in analysis (Primary, Secondary, Tertiary, University, ostgraduate)', default=None)
parser.add_argument('--export', metavar='FILENAME', help='Instead of running KNN, export numpy arrays with CLIP vectors and scores to the given file.', default=None)
args = parser.parse_args()

def log(s, level=1, flush=False):
    if args.quiet and level > 0: return
    print(s, flush=flush)

if args.random_seed: seed(args.random_seed)

# The various categories of ratings in our survey
categories = ['walkability', 'bikeability', 'pleasantness', 'greenness', 'safety']

clipmodelname = args.clip_model

# A filename-friendly version of the clip-model name:
clipmodelfilepart = args.clip_model.replace('/', '_')

# We will store embeddings in (or load them from) the dir in 'embdir'
if args.embeddings_dir is None:
    embdir = 'embeddings-' + clipmodelfilepart
else:
    embdir = args.embeddings_dir 

if args.environmental:
    # If complementary environment variables are being used then prepare a
    # directory to store text prompts and their associated embeddings.
    assert(args.prompt_style in [0, 1])
    textembdir = 'text-' + f'prompt{args.prompt_style}-' + embdir
    if args.environmental_text_dir is not None:
        textdir = Path(args.environmental_text_dir)
    else:
        textdir = Path(textembdir) / 'text/'
    textdir.mkdir(parents=True, exist_ok=True)
    textnpyfile = Path(textembdir) / 'text_emb/text_emb_0.npy'
    textmetadatafile = Path(textembdir) / 'metadata/metadata_0.parquet'

if args.images_dir is not None and not Path(embdir).exists():
    # If embeddings don't yet exist then run the clip-retrieval tool to efficiently generate them
    cmd=["clip-retrieval", 'inference', '--input_dataset', args.images_dir, '--output_folder',  embdir, '--clip_model', args.clip_model] + args.other_clip_retrieval_args.split(' ')
    log(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)

metadatafile = Path(embdir) / 'metadata/metadata_0.parquet'
npyfile = Path(embdir) / 'img_emb/img_emb_0.npy'

if not metadatafile.exists():
    log(f'Cannot find metadata file {str(metadatafile)}, aborting.', level=0, flush=True)
    sys.exit(1)
if not npyfile.exists():
    log(f'Cannot find numpy saved array file {str(npyfile)}, aborting.', level=0, flush=True)
    sys.exit(1)
if not Path(args.geojson).exists():
    log(f'Cannot find GeoJSON file {args.geojson}, aborting.', level=0, flush=True)
    sys.exit(1)

# Load the clip-retrieval metadata, which in the case of image embeddings, is
# simply an array of image filenames corresponding to the vectors found in the
# array stored in the npyfile.
table = pq.read_table(metadatafile)[0]
num_images = table.length()
log(f'Read metadata for {num_images} images.')

# Load image CLIP vector data
allvecs = np.load(npyfile)
allvecs = np.asarray(allvecs, np.float64)
allvecs = allvecs / np.expand_dims(np.linalg.norm(allvecs, axis=1), axis=1)

# fields in the demographic data that should be coerced to type int.
int_fields = ['image_id', 'category_id', 'rating_id', 'session_id', 'rating', 'score', 'age']

# Open the demographics file if given, store the entries in the 'demodb' table,
# indexed by image_id.
demodb = {}
if args.demographics is not None:
    with open(args.demographics) as fp:
        reader = csv.reader(fp, delimiter=',')
        fields = reader.__next__()
        for row in reader:
            entry = {}
            for field, col in zip(fields,row):
                if field in int_fields:
                    entry[field] = int(col)
                else:
                    entry[field] = col
            imgid = entry['image_id']
            if imgid not in demodb: demodb[imgid] = []
            demodb[imgid].append(entry)
    log(f'Read demographics file {args.demographics} and recorded {len(demodb.keys())} image_id entries.')

# Filter by gender: --gender='man,woman,non-binary,...' or some subset
gender_filter = None
if args.demographics and args.gender:
    gender_filter = args.gender.split(',')
    print(f'Filtering by surveyed gender: {gender_filter}')

# Region filter: supports --region=NL or --region=non-NL
# This uses both country and postcode to try and establish the region of the participant.
# FIXME: the regexes are based on our existing dataset, with its peculiarities
# encoded here; for more general use either you need to clean up your dataset
# before running it with this code or modify this code to handle your specific
# cases.
nl_country_regex = regex.compile('^NL$|Nederland|Netherlands|Dutch|Utrecht|Amsterdam|netherland|netjerlands|Netherl')
nl_postcode_regex = regex.compile('^[0-9]{4} *[A-Z]{2}$')

def nl_filter(e):
    c = e['country']
    p = e['postalcode']
    return nl_country_regex.match(c) or nl_postcode_regex.match(p)

region_filter = lambda x: True
if args.region == 'NL':
    region_filter = nl_filter
    print('Filtering by region: NL')
elif args.region == 'non-NL':
    region_filter = lambda x: not nl_filter(x)
    print('Filtering by region: non-NL')
elif args.region is not None: log(f'Invalid region: {args.region}', level=0)

# Age filter:
#   if --age=n then it looks for people in the age range [n,120] (inclusive)
#   if --age=n,m then it looks for people in the age range [n,m] (inclusive)
age_filter = None
if args.age is not None:
    ages = args.age.split(',')
    if len(ages) == 1:
        age_filter = [int(ages[0]), 120]
    elif len(ages) == 2:
        age_filter = [ int(ages[0]), int(ages[1]) ]
    else:
        log(f'Invalid age filter: {args.age}', level=0)

# Education level filter: --education='Primary,Secondary,Tertiary,Postgraduate' or some subset
edu_filter = None
if args.education is not None:
    edu_filter = args.education.split(',')
    log(f'Filtering by surveyed education level: {edu_filter}')


# Given a particular demographic profile 'e' decide whether or not it belongs
# in the data sample based on the filters set up by command-line arguments.
def filter_by_demographics(e):
    if gender_filter is not None:
        if e['gender'] not in gender_filter: return False
    if not region_filter(e): return False
    if age_filter is not None:
        if e['age'] < age_filter[0] or e['age'] > age_filter[1]: return False
    if edu_filter is not None:
        if e['education'] not in edu_filter: return False
    # other filters go here

    return True

# Load the main GeoJSON file as JSON data
with open(args.geojson) as fp:
    js = json.load(fp)

# Populate the main 'db' table with entries indexed by image filename and
# containing values with all the properties found in the GeoJSON file.
# (FIXME: rewrite this using (Geo)Pandas)
db = {}
for feat in js['features']:
    props = feat['properties']
    url = props['url']
    fn = url[url.rfind('/')+1:]
    db[fn] = props


# Generate environmental text strings if requested
if args.environmental:
    # List of column names pertaining to complementary environmental data
    env_keys = ['greenspace_count_100', 'shops_count_100', 'public_transport_count_100', 'sustenance_count_100', 'education_count_100', 'financial_count_100', 'healthcare_count_100', 'entertainment_count_100', 'greenspace_density_or_proportion_100', 'shops_density_or_proportion_100', 'public_transport_density_or_proportion_100', 'sustenance_density_or_proportion_100', 'education_density_or_proportion_100', 'financial_density_or_proportion_100', 'healthcare_density_or_proportion_100', 'entertainment_density_or_proportion_100', 'greenspace_shannon_entropy_100', 'shops_shannon_entropy_100', 'public_transport_shannon_entropy_100', 'sustenance_shannon_entropy_100', 'education_shannon_entropy_100', 'financial_shannon_entropy_100', 'healthcare_shannon_entropy_100', 'entertainment_shannon_entropy_100', 'bike_network_length_100', 'walk_network_length_100', 'num_nodes_100', 'num_edges_100', 'streets_per_node_avg_100', 'streets_per_node_proportions1_100', 'streets_per_node_proportions3_100', 'streets_per_node_proportions4_100', 'intersection_count_100', 'street_length_total_100', 'street_segment_count_100', 'street_length_avg_100', 'orientation_entropy_100', 'median_speed_100', 'greenspace_count_300', 'shops_count_300', 'public_transport_count_300', 'sustenance_count_300', 'education_count_300', 'financial_count_300', 'healthcare_count_300', 'entertainment_count_300', 'greenspace_density_or_proportion_300', 'shops_density_or_proportion_300', 'public_transport_density_or_proportion_300', 'sustenance_density_or_proportion_300', 'education_density_or_proportion_300', 'financial_density_or_proportion_300', 'healthcare_density_or_proportion_300', 'entertainment_density_or_proportion_300', 'greenspace_shannon_entropy_300', 'shops_shannon_entropy_300', 'public_transport_shannon_entropy_300', 'sustenance_shannon_entropy_300', 'education_shannon_entropy_300', 'financial_shannon_entropy_300', 'healthcare_shannon_entropy_300', 'entertainment_shannon_entropy_300', 'bike_network_length_300', 'walk_network_length_300', 'num_nodes_300', 'num_edges_300', 'streets_per_node_avg_300', 'streets_per_node_proportions1_300', 'streets_per_node_proportions3_300', 'streets_per_node_proportions4_300', 'intersection_count_300', 'street_length_total_300', 'street_segment_count_300', 'street_length_avg_300', 'orientation_entropy_300', 'median_speed_300']

    # Read the GeoJSON info (again) into a GeoDataFrame to work with it Pandas-style
    # (FIXME: consolidate this with the previous JSON-based read of the same file)
    gdf = gpd.read_file(args.geojson)

    # qgdf will be a copy of the dataframe with 'quintile' columns added for each environmental variable
    qgdf = gdf.copy()
    for col in env_keys:
        # Find the 'unique values' possible in this data column. We work with
        # unique values because we are interested in the range of possible
        # values and not so much the distribution. E.g., most roads have a
        # speed limit of 50 km/h (the default), a large number are also 30
        # km/h, a few are 20 km/h and a bunch are also 90 km/h or higher. If we
        # went based on distribution then it is likely that the 40%, 60% and
        # 80% percentile speeds would all be 50 km/h, which isn't terribly
        # useful when categorizing the range of possible speed limits.
        uv = gdf[col].unique()
        # divide the unique values into quintiles (quantiles with divisions at 20%, 40%, ...)
        qt = np.quantile(uv, [0.2, 0.4, 0.6, 0.8])
        if len(uv) >= 5:
            # pd.cut will return a column that categorizes the input values
            # into the various 'bins' we provide, which in this case are
            # defined by the quintiles (bracketed by -1 and max+1, which both
            # lie outside the range of possible values), and we label each bin
            # with a number, 1 to 5.
            qgdf[f'{col}_quintile'] = pd.cut(gdf[col], bins=[-1] + qt.tolist() + [np.max(uv)+1], labels=[1, 2, 3, 4, 5], include_lowest=True, duplicates='drop')
        else:
            # Do the same but since len(uv) < 5 then we will use fewer bins.
            # FIXME: this needs testing, it doesn't come up in the current data
            # set because len(uv) >= 6 in all cases so far
            qgdf[f"{col}_quintile"] = pd.cut(gdf[col], bins=[-1] + sorted(uv) + [np.max(uv)+1], labels=list(range(1, len(uv)+1)), include_lowest=True)

    # Pretty print the name of the column/key, replacing _ with space,
    # extracting the buffer size and formatting it nicely, and special-casing a
    # few names.
    def reformat(k):
        words = k.split('_')
        bufsize = words[-1]
        if words[:-1] == ['streets', 'per', 'node', 'proportions1']:
            name = 'number of cul-de-sacs'
        elif words[:-1] == ['streets', 'per', 'node', 'proportions3']:
            name = 'number of 3-way intersections'
        elif words[:-1] == ['streets', 'per', 'node', 'proportions4']:
            name = 'number of 4-way intersections'
        elif words[:-1] == ['num', 'nodes']:
            name = 'number of intersections'
        elif words[:-1] == ['num', 'edges']:
            name = 'number of street segments'
        else:
            name = " ".join(words[:-1])
        return f'{name} (within buffer of size {bufsize}m)'

    # Check if the text CLIP vectors already exist
    if not textnpyfile.exists():
        if args.environmental_text_dir is None:
            # Generate prompts that correspond to each of the locations in the data.
            for fn, props in db.items():
                textfn = textdir / Path(fn).with_suffix('.txt')
                # Do not regenerate the prompt if it was already written to file.
                if textfn.exists(): continue
                def to_quintile(k):
                    # Look up the quintile ranking in qgdf and convert it to a text description.
                    q = qgdf.loc[qgdf['image_id'] == props['image_id']][f'{k}_quintile'].item()
                    return ['very low', 'low', 'medium', 'high', 'very high'][q-1]

                if args.prompt_style == 0:
                    # Prompt-style 0: just output the raw quantity
                    text = '; '.join([f'{reformat(k)} is {v}' for k, v in props.items() if k in env_keys])
                elif args.prompt_style == 1:
                    # Prompt-style 1: output the quintile for the quantity, pretty-printed as 'very low', 'low', etc.
                    text = '; '.join([f'{reformat(k)} is {to_quintile(k)}' for k, v in props.items() if k in env_keys])
                #log(f"{textfn}: {text}")
                with open(textfn, 'w') as fp:
                    fp.write(text + '\n')

        # Invoke the clip-retrieval tool to generate text embeddings
        cmd=["clip-retrieval", 'inference', '--input_dataset', str(textdir), '--output_folder', str(textembdir), '--clip_model', args.clip_model] + args.other_clip_retrieval_args.split(' ')
        log(f'Running: {" ".join(cmd)}')
        subprocess.run(cmd, check=True)
        # These embeddings will be found in the path contained in 'textnpyfile' if successful
        if not textnpyfile.exists():
            log(f'clip-retrieval failed: cannot find {str(textnpyfile)}', level=0)
            sys.exit(1)

    # Load the text CLIP vectors
    alltextvecs = np.load(textnpyfile)
    texttable = pq.read_table(textmetadatafile)[0]
    num_texts = texttable.length()
    alltextvecs = np.asarray(alltextvecs, np.float64)
    alltextvecs = alltextvecs / np.expand_dims(np.linalg.norm(alltextvecs, axis=1), axis=1)
    log(f'Read metadata for {num_texts} texts.')

    # Assumption: alltextvecs are in the same order as allvecs, because the
    # filenames are the same (apart from the .txt ending); unfortunately the
    # metadata produced by clip-retrieval does not provide the filenames of
    # text files that are encoded, only the contents.
    #
    # The following code checks that text encoding vectors are in the same
    # order as the image encoding vectors.
    for i, pathname in enumerate(table):
        pathname = str(pathname)
        fn = pathname[pathname.rfind('/')+1:]
        textfn = textdir / Path(fn).with_suffix('.txt')
        with open(textfn) as fp:
            text = fp.read().strip()
        if text != str(texttable[i]).strip():
            log(f'Text inconsistency for entry {i}: {textfn}', level=0)
            log(f'Text from metadata table entry {i}: {texttable[i]}', level=0)
            log(f'Text from file {textfn}: {text}', level=0)
            sys.exit(1)

    args.environmental_method = args.environmental_method.lower()
    if args.environmental_method == 'append':
        # For each (img vector, textvector) pair, output [img vector..., text vector...]
        allvecs = np.append(allvecs, alltextvecs, axis=1)
    elif args.environmental_method == 'average':
        # For each (img vector, textvector) pair, output [average(img vector, text vector)...]
        if alltextvecs.shape != allvecs.shape:
            log(f'Shape mismatch: alltextvecs.shape ({alltextvecs.shape}) != allvecs.shape ({allvecs.shape})', level=0)
            sys.exit(1)
        allvecs = np.average([allvecs, alltextvecs], axis=0)
    elif args.environmental_method == 'slerp':
        # For each (img vector, textvector) pair, output [slerp(img vector, text vector, 0.5)...]
        if alltextvecs.shape != allvecs.shape:
            log(f'Shape mismatch: alltextvecs.shape ({alltextvecs.shape}) != allvecs.shape ({allvecs.shape})', level=0)
            sys.exit(1)
        # slerp works on float64s and gets picky if the vectors are not very close to unit-length
        slerp_interval = 0.5
        # slerp is not vectorized
        for i in range(allvecs.shape[0]):
            allvecs[i] = geometric_slerp(allvecs[i], alltextvecs[i], slerp_interval)
    else:
        log(f'Invalid --environmental-method: {args.environmental_method}', level=0)
        sys.exit(1)


log(f'allvecs.shape = {allvecs.shape}')

# Loop through the image vector metadata, which is a list of image pathnames
# that were encoded into vectors now stored in the array allvecs.
for i, pathname in enumerate(table):
    pathname = str(pathname)
    fn = pathname[pathname.rfind('/')+1:]
    if fn not in db:
        log(f'Did not find image {fn} in GeoJSON file', level=0)
        continue

    props = db[fn]
    if demodb:
        # demographic-filtering is enabled
        imgid = props['image_id']
        # information about the demographics of survey participants is stored
        # in the table 'demodb', which is indexed by image_id. The values of
        # the 'demodb' table are each a list of entries, corresponding to
        # individual ratings given by individual participants.
        for entry in demodb[imgid]:
            cat = categories[entry['category_id']-1]
            if cat not in db: db[cat] = { 'scores': [], 'vecs': [], 'imgids': [] }
            # Look at the demographics of the individual participant who gave
            # this rating and decide whether or not to filter them out.
            if filter_by_demographics(entry):
                # We decided to include this rating, so add this particular
                # (rating, image_id, CLIP vector) triple to the main table,
                # indexed by category:
                db[cat]['scores'].append(entry['rating'])
                db[cat]['imgids'].append(imgid)
                db[cat]['vecs'].append(allvecs[i])
    else:
        # Do not use demographic filtering. The GeoJSON file contains average
        # ratings in each category for each image, averaged across every
        # participant who rated a given image in that category.
        for cat in categories:
            lbl = f'average_{cat}'
            if cat not in db: db[cat] = { 'scores': [], 'vecs': [], 'imgids': [] }
            if lbl in props and props[lbl] is not None:
                # Add the (average_rating, image_id, CLIP vector) triple to the
                # main table, indexed by category:
                db[cat]['scores'].append(props[lbl])
                db[cat]['imgids'].append(props['image_id'])
                db[cat]['vecs'].append(allvecs[i])


# Export-mode: dump the vectors and scores into an NPZ file
if args.export is not None:
    # Run KNN analysis category by category
    out = {}
    for cat in categories:
        d = db[cat]
        log(f"Found {len(d['imgids'])} ratings in category {cat}.")
        out[f'{cat}_scores'] = np.array(d['scores'], dtype=float)
        out[f'{cat}_vecs'] = np.array(d['vecs'], dtype=allvecs.dtype)

    log(f'Saving to file {args.export}.')
    np.savez_compressed(args.export, **out)
    log('Export complete.')

    sys.exit(0)

# Run KNN analysis category by category
for cat in categories:
    d = db[cat]
    log(f"Found {len(d['imgids'])} ratings in category {cat}.")
    d['scores'] = np.array(d['scores'], dtype=float)
    d['imgids'] = np.array(d['imgids'], dtype=int)
    d['vecs'] = np.array(d['vecs'], dtype=allvecs.dtype)


    if args.stratified:
        # Split into stratified train/test sets & randomize order if requested.
        # Stratify by score (1 to 5) so that the training and testing sets have
        # similar proportions of each score.
        #
        # Split the indices of the d['scores'] array into 5 bins
        bins = [np.where(d['scores'].astype(int) == i)[0] for i in range(1, 6)]
        # Find the number of training samples in each bin
        traincounts = [int(float(bins[i].size) * args.training_split) for i in range(5)]
        testcounts  = [bins[i].size - traincounts[i] for i in range(5)]
        if args.randomize:
            # Shuffle each bin separately if randomization is requested
            for i in range(5): shuffle(bins[i])
        # Training and testing split for each bin
        trainbininds = [bins[i][:c] for (i, c) in enumerate(traincounts)]
        testbininds  = [bins[i][c:] for (i, c) in enumerate(traincounts)]
        traininds = np.concatenate(trainbininds)
        testinds  = np.concatenate(testbininds)
        if args.extra_assertions:
            # Array sizes should all add up to the original bins
            assert (np.all([bins[i].shape[0] == trainbininds[i].shape[0] + testbininds[i].shape[0] for i in range(5)]))
            # Training set and testing set must not overlap
            assert (np.intersect1d(traininds, testinds).size == 0)
        log(f'Stratified splits for {cat}: |training sets| = {list(traincounts)}; |testing sets| = {list(testcounts)}')
        # for later use in logging
        traincount = '+'.join(map(str,traincounts))
        testcount  = '+'.join(map(str,testcounts))
    else:
        # Split into train/test sets & randomize order if requested.
        count = d['vecs'].shape[0]
        traincount = int(float(count) * args.training_split)
        testcount = count - traincount
        log(f'Split for {cat}: |training set| = {traincount}; |testing set| = {testcount}')
        inds = np.arange(count)
        if args.randomize:
            shuffle(inds)
        traininds = inds[:traincount]
        testinds  = inds[traincount:]

    d['trainscores'] = d['scores'][traininds]
    d['testscores'] = d['scores'][testinds]
    d['trainimgids'] = d['imgids'][traininds]
    d['testimgids'] = d['imgids'][testinds]
    d['trainvecs'] = d['vecs'][traininds]
    d['testvecs'] = d['vecs'][testinds]


    (num_vec_rows, clipvecsize)       = d['vecs'].shape
    (num_trainvec_rows, clipvecsize1) = d['trainvecs'].shape
    (num_testvec_rows, clipvecsize2)  = d['testvecs'].shape

    # Compute the cosine similarities between all 'training set' vectors and
    # all 'testing set' vectors. This is, in essence, just a dot product for
    # each vector combination, divided by the magnitudes of both vectors (to
    # normalize it). En masse, this can be computed with a single matrix
    # multiplication followed by two vector divisions, taking care to ensure
    # that the vectors are arranged properly.
    trainnorms = np.expand_dims(np.linalg.norm(d['trainvecs'], axis=1), axis=0)
    testnorms  = np.expand_dims(np.linalg.norm(d['testvecs'], axis=1), axis=1)
    if args.extra_assertions:
        assert (clipvecsize == clipvecsize1 and clipvecsize == clipvecsize2 and
                num_trainvec_rows == trainnorms.shape[1] and num_testvec_rows == testnorms.shape[0] and
                num_trainvec_rows + num_testvec_rows == num_vec_rows)
    cos_sim_table = d['testvecs'] @ d['trainvecs'].T / trainnorms / testnorms
    # cos_sim_table is a matrix (i, j) representing the similarity between
    # testing vector i and training vector j.
    d['cos_sim_table'] = cos_sim_table

    # Sort the cosine similarity table, for each testing vector find the order
    # of training vectors that have the greatest similarity to th least
    # similarity (hence use of np.flip on axis=1); track this sortation using
    # indices so that we can reuse those indices for look-ups in other tables.
    sortinds = np.flip(np.argsort(cos_sim_table, axis=1), axis=1)
    if args.extra_assertions:
        # Just check that the 0th (best) training vector per testing vector
        # does indeed have the max similarity.
        max_similarity_per_testvec            = np.max(cos_sim_table, axis=1)
        best_trainvec_index_per_testvec       = sortinds[:, 0]
        best_trainvec_similiarity_per_testvec = \
            cos_sim_table[np.arange(num_testvec_rows), best_trainvec_index_per_testvec]
        assert (np.all(max_similarity_per_testvec == best_trainvec_similiarity_per_testvec))

    # Run K-nearest neighbour with a given value of K
    def do_k(k):
        # kscores is the matrix of (average) ratings associated with the top-K
        # similar training vectors, for each of the testing vectors.
        kscores = d['trainscores'][sortinds][:,:k]
        # Use the sortinds matrix to actually index into cos_sim_table
        ksims = cos_sim_table[np.expand_dims(np.arange(sortinds.shape[0]), axis=1), sortinds]
        if args.extra_assertions:
            # Check that ksims is in fact the sorted cos_sim_table (along axis=1)
            assert (np.all(ksims == np.flip(np.sort(cos_sim_table), axis=1)))
        ksims = ksims[:,:k]
        # ksims is now the matrix of similarities associated with the top-K
        # similar training vectors, for each of the testing vectors.
        if args.extra_assertions:
            assert (ksims.shape == (num_testvec_rows, k) and kscores.shape == (num_testvec_rows, k))

        # Normalize the ksims row-by-row such that each row adds up to 1 (thus
        # forming a normalized vector for weighting the kscores). However,
        # there are a number of methods to do this, so we make it a
        # command-line option.
        if args.normalization_method == 'divbysum':
            ksims = ksims / np.expand_dims(np.sum(ksims, axis=1), axis=1)
        elif args.normalization_method == 'softmax**8':
            ksims = softmax(ksims**8)
        elif args.normalization_method == 'softmax10**':
            ksims = softmax(10**ksims)
        elif args.normalization_method == 'fixed':
            # Just produce a fixed list of integers with which to weight the ksims
            ksims = np.repeat(np.expand_dims(np.flip(np.array(range(k))),axis=0),[ksims.shape[0]],axis=0)
            ksims = softmax(ksims)
        else:
            ksims = softmax(ksims)

        # At the end, we have a weighted score for each testing vector
        kweightedscores = np.sum(kscores * ksims, axis=1)

        # yg = 'Y' values for ground truth
        yg = d['testscores']
        # yp = 'Y' values for predictions
        yp = kweightedscores

        mse = mean_squared_error(yg, yp)
        r2  = r2_score(yg, yp)

        log(f'category = {cat}; k = {k}; mse = {mse:.05f}; r2  = {r2:.05f}', level=0, flush=True)
        if args.results_log is not None:
            with open(args.results_log, 'a') as fp:
                csvw = csv.writer(fp)
                rand = 'random=' + ((str(args.random_seed) if args.random_seed else 'True') if args.randomize else 'False')
                strat = 'stratified=' + ('True' if args.stratified else 'False')
                env = 'env=None' if not args.environmental else f'env={args.environmental_method}+p{args.prompt_style}'
                csvw.writerow([cat, clipmodelname, k, rand, strat, traincount, testcount, mse, r2, args.normalization_method, env])

    # Handle the case where k is a comma-separated list of k values
    if type(args.k) == str:
        for k in [int(k) for k in args.k.split(',')]:
            do_k(k)
    elif type(args.k) == int:
        do_k(args.k)
    else:
        log(f'Unrecognized value for k={args.k}', level=0)
