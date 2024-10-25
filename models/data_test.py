import models.iter_dataset as iter_dataset
import unittest
import torch
import subprocess
import re
import json
from pathlib import Path
from typing import Union

#Number of sample extracted for every test
SAM:int = 2000

# Check dataset func without info file
def no_info(dir, sam:int = SAM):
    if not isinstance(dir, Path):
        supp = Path(dir) / ".info.json"
    else:
        supp = dir / ".info.json"
    supp.unlink(missing_ok=True)
    ds = iter_dataset.DirectoryRandomDataset(dir)
    iter_test(ds, sam)
    print("No info passed")

#Check image
def check_image(img):
    assert isinstance(img, torch.Tensor)
    shape = img.shape
    assert len(shape) == 4
    assert shape[0] == 1 and shape[1] == 3 and shape[2] != 0 and shape[3] != 0
    assert img.dtype == torch.float32

#Try to iterae over sam element
def iter_test(ds, sam:int):
    it = ds.__iter__()
    for i in range(sam):
        img, lab = next(it)
        check_image(img)
        assert isinstance(lab, torch.Tensor)
        assert lab == 0 or lab == 1
        if((i + 1) % 100 == 0):
            print(f"Test {i + 1} passed!")
    assert i == sam - 1

#Check dataset functionality with info file
def with_info(dir, sam:int = SAM):
    if not isinstance(dir, Path):
        supp = Path(dir)
    ds = iter_dataset.DirectoryRandomDataset(dir)
    assert ds.len > 0
    iter_test(ds, sam)
    print("With info passed")

#Check dataset func with true mode
def true_test(dir, sam:int = SAM):
    ds = iter_dataset.DirectoryRandomDataset(dir)
    it = ds.__iter__()
    ds.change_mode(ds.REAL)
    for i in range(sam):
        img, lab = next(it)
        check_image(img)
        assert isinstance(lab, torch.Tensor)
        assert lab == 0
        if((i + 1) % 100 == 0):
            print(f"Test {i + 1} passed!")
    print("True Test passed")

#Check dataset func with fake mode
def fake_test(dir, sam:int = SAM):
    ds = iter_dataset.DirectoryRandomDataset(dir)
    it = ds.__iter__()
    ds.change_mode(ds.FAKE)
    for i in range(sam):
        img, lab = next(it)
        check_image(img)
        assert isinstance(lab, torch.Tensor)
        assert lab == 1
        if((i + 1) % 100 == 0):
            print(f"Test {i + 1} passed!")
    print("Fake Test passed")

#Check dataset func with couple mode
def coup_test(dir, sam:int = SAM):
    ds = iter_dataset.DirectoryRandomDataset(dir)
    it = ds.__iter__()
    ds.change_mode(ds.COUP)
    for i in range(sam):
        img1, img2 = next(it)
        check_image(img1)
        check_image(img2)
        if((i + 1) % 100 == 0):
            print(f"Test {i + 1} passed!")
    print("Couple Test passed")

#Check dataset func with wrong directory
def fake_dir(dir:Union[str, Path]):
    try:
        ds = iter_dataset.DirectoryRandomDataset(dir)
    except ValueError as e:
        assert e.__str__() == "The path have to be a directory"
        print("Fake dir passed")
        return
    assert False

#Check dataset func with wrong format
def wrong_form(dir):
    try:
        ds = iter_dataset.DirectoryRandomDataset(dir, ext="alaksa")
        next(ds.__iter__())
    except RuntimeError as e:
        assert e.__str__() == "Image not present, some problem occur"
        print("Wrong format passed")
        return
    assert False

#Check the info file correcteness
def correct_info(dir):
    if not isinstance(dir, Path):
        supp = Path(dir)
    else:
        supp = dir
    with open(supp / ".info.json") as f:
        dlen = json.load(f)["len"]
    assert dlen % 2 == 0
    out = subprocess.check_output(f"ls -1 {supp.resolve()} | wc -l", shell=True)
    assert dlen == int(re.findall(r"\d+", str(out))[0])
    ds = iter_dataset.DirectoryRandomDataset(dir, check=True)
    assert dlen // 2 == ds.len
    print("Correct info")

def random_test(dir:Union[str, Path]):
    no_info(dir)
    correct_info(dir)
    with_info(dir)
    true_test(dir)
    fake_test(dir)
    coup_test(dir)
    fake_dir("//not_a_dir_ago//")
    fake_dir("//homes//dagostini//CVCSproj//model//data_test.py")
    wrong_form(dir)

if __name__ == "__main__":
    random_test("//work//cvcs2024//VisionWise//test")
    random_test(Path("//work//cvcs2024//VisionWise//test"))
    print("All test passed")
