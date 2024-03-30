"""
FILE_NAME: tab_mgr.py
AUTHOR: RCB
CREATED: 2024-01-14-12:01
DESC: 标签页面管理器
"""
from .tab_base import TabBase


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]


@Singleton
class TabMgr(object):
    def __init__(self):
        self._tabs = {}

    def registerTab(self, tab_obj: TabBase):
        if tab_obj.tab_name:
            self._tabs[tab_obj.tab_name] = tab_obj

    def clearTab(self):
        self._tabs = {}

    def getTab(self, tab_name):
        if tab_name in self._tabs:
            return self._tabs[tab_name]
        return None

    def implUILogic(self):
        for _, tab_obj in self._tabs.items():
            tab_obj.impl_ui_logic()


def get_tab_mgr() -> TabMgr:
    return TabMgr()
