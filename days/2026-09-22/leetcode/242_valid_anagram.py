"""
242. 有效的字母异位词  https://leetcode.cn/problems/valid-anagram/
9/21 计划题。没做就今天先补这一道，再做 128。

给定两个字符串 s 和 t，判断 t 是否是 s 的字母异位词。
异位词：字母相同、顺序可以不同。只含小写字母。

自己写。不要 import collections 以外的库。
"""


def isAnagram(s: str, t: str) -> bool:
    raise NotImplementedError


def _check():
    assert isAnagram("anagram", "nagaram") is True
    assert isAnagram("rat", "car") is False
    assert isAnagram("a", "a") is True
    assert isAnagram("ab", "a") is False
    print("242 ok")


if __name__ == "__main__":
    _check()
