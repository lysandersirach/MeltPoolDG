#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:03:01 2021

@author: magdalena
"""

import json
import collections
from functools import reduce
from operator import getitem
import os

import collections.abc


##############################################################################
#          USER INPUT
##############################################################################

old_parameter_names = [
    ["level set", "reinitialization", "hyperbolic", "interface thickness parameter"],
    ["level set", "reinitialization", "geometric", "interface thickness parameter"],
    ["reinitialization", "hyperbolic", "interface thickness parameter"],
    ["reinitialization", "geometric", "interface thickness parameter"],
    # ... add old parameter names
    # ["old", "my age"],
]
new_parameter_names = [
    ["level set", "reinitialization", "interface thickness parameter"],
    ["level set", "reinitialization", "interface thickness parameter"],
    ["reinitialization", "interface thickness parameter"],
    ["reinitialization", "interface thickness parameter"],
    # ... add new parameter names
    # ["new", "new", "my new age"],
]
delete_parameter_names = [
    # ... add parameter names to be deleted
]

new_categories = [
    # {'new': {'new': {'new': {}}}},
]  # optional: add entirely new parameter categories

root_dir = "."

##############################################################################
#         END OF USER INPUT
##############################################################################

# collect json files
json_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(
    root_dir, "tests")) for f in filenames if os.path.splitext(f)[1] == '.json']
json_files += [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(
    root_dir, "applications")) for f in filenames if os.path.splitext(f)[1] == '.json']


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_nested_value(dataDict, mapList):
    """Get item value in nested dictionary"""
    return reduce(getitem, mapList[:-1], dataDict)[mapList[-1]]


def set_nested_item(dataDict, mapList, val):
    """Set item in nested dictionary"""
    reduce(getitem, mapList[:-1], dataDict)[mapList[-1]] = val
    return dataDict


def delete_nested_item(dataDict, mapList, val):
    """Delete item in nested dictionary"""
    del reduce(getitem, mapList[:-1], dataDict)[mapList[-1]]


def is_empty(value):
    return value is None or value == "" or value == [] or value == {}


def remove_empty_items(obj):
    if isinstance(obj, dict):
        cleaned = {
            key: remove_empty_items(value)
            for key, value in obj.items()
        }
        return {
            key: value
            for key, value in cleaned.items()
            if not is_empty(value)
        }

    if isinstance(obj, list):
        cleaned = [remove_empty_items(item) for item in obj]
        return [item for item in cleaned if not is_empty(item)]

    return obj


def remove_empty_nested_items(d):
    for i in range(10):  # max 10 nested parameters
        d = remove_empty_items(d)
    return (d)


def modify_json(json_f, appendix=""):
    assert len(json_f) > 0
    assert len(old_parameter_names) == len(new_parameter_names)

    for j in json_f:
        with open(j, 'r') as f:
            print(70 * "-")
            print("Process file: {:}".format(j))
            datastore = json.load(f, object_pairs_hook=collections.OrderedDict)

            val = datastore

            # process parameters to be renamed
            for i, o in enumerate(old_parameter_names):
                # get value of old parameter
                try:
                    val = get_nested_value(datastore, o)
                except BaseException:
                    print("    Old value {:} not found. Abort...".format(o))
                    continue
                # introduce new parameter category
                for cat in new_categories:
                    update(datastore, cat)
                    print("    New category introduced: {:}".format(cat))
                # introduce new parameter name and set value from the old parameter
                set_nested_item(datastore, new_parameter_names[i], val)
                # delete the outdated parameter pair
                delete_nested_item(datastore, o, val)
                print("    Rename OLD {:} to NEW {:}".format(
                    o, new_parameter_names[i]))

            # process parameters to be deleted
            for i, o in enumerate(delete_parameter_names):
                try:
                    val = get_nested_value(datastore, o)
                except BaseException:
                    print("Value {:} not found.".format(o))
                    continue
                delete_nested_item(datastore, o, val)
                print("Delete parameter '{:}'".format(o))

            # delete potentially empty items
            datastore = remove_empty_nested_items(datastore)

        with open(j.split(".json")[0] + appendix + ".json", 'w') as f:
            json.dump(datastore, f, indent=2, separators=(',', ': '))


if __name__ == "__main__":
    modify_json(json_files)

    # DEBUG: test
    # modify_json(["modify_json.json"], "new")
