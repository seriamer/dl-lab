"""
128. 最长连续序列  https://leetcode.cn/problems/longest-consecutive-sequence/
今天必做。要求 O(n) 时间，所以排序不算过关。

给定未排序整数数组，返回最长连续序列的长度。
连续指数值连续，例如 [1,2,3,4]，不要求下标连续。

思路：放进 set。对每个 x，只有 x-1 不在 set 里时，x 才是一段的起点，
然后向右数 x+1, x+2, ... 直到断。每个数最多被访问常数次。
"""


def longestConsecutive(nums: list[int]) -> int:
    raise NotImplementedError


def _check():
    assert longestConsecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    assert longestConsecutive([]) == 0
    assert longestConsecutive([1]) == 1
    assert longestConsecutive([1, 2, 0, 1]) == 3
    print("128 ok")


if __name__ == "__main__":
    _check()
