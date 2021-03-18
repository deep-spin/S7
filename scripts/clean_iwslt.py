#!/usr/bin/env python

import xml.etree.ElementTree as ET
import argparse
import os
from os.path import join, basename
from glob import glob
from collections import defaultdict
from itertools import chain


def train_lines(path):
    xml_tags = ['<url', '<keywords', '<talkid', '<description',
                '<reviewer', '<translator', '<title', '<speaker']
    with open(path) as f:
        for line in f:
            if not any(tag in line for tag in xml_tags):
                yield line.strip()


def dev_lines(path):
    root = ET.parse(path, parser=ET.XMLParser(encoding='utf-8')).getroot()[0]
    for doc in root.findall('doc'):
        for e in doc.findall('seg'):
            yield e.text.strip()


def dev_name(path):
    return ".".join(basename(path).split(".")[:-2])


def dev_paths(dirname):
    paths = defaultdict(list)
    for dev_path in glob(join(dirname, "*.xml")):
        lang = dev_path.split(".")[-2]
        paths[lang].append(dev_path)
    for k in paths:
        paths[k].sort()
    assert all(dev_name(f2) == dev_name(f2) for f1, f2 in zip(*paths.values()))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("out_directory")
    opt = parser.parse_args()
    os.makedirs(opt.out_directory, exist_ok=True)
    for path in glob(join(opt.directory, "train.tags.*")):
        lang = path.split(".")[-1]
        lines = train_lines(path)
        with open(join(opt.out_directory, "train." + lang), "w") as f:
            for line in lines:
                f.write(line + "\n")
    
    for lang, paths in dev_paths(opt.directory).items():
        lines = chain.from_iterable([dev_lines(p) for p in paths])
        with open(join(opt.out_directory, "valid." + lang), "w") as f:
            for line in lines:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
