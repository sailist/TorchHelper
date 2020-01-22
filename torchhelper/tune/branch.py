'''
   Copyright 2020 Sailist

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
from collections import OrderedDict, Iterable

class PseudoVar:
    def __getattr__(self, item):
        return item

class VarDict(dict):
    def __getattr__(self, item):
        return self[item]

class FuncGraph:
    """
    模仿TensorFlow的静态图构造出的静态执行方法的类

    ::
        def add(a, b):
        print("add")
        return a + b

        def multi(a, b):
            print("multi")
            return a * b

        def swap(a, b):
            print("swap")
            return b, a

        func = FuncGraph()
        p = PseudoVar()
        func.add_func_node(add, [p.a, p.b], to=[p.c])
        func.add_func_node(multi, [p.a, p.c], to=[p.d])
        func.add_func_node(swap, [p.a, p.b], to=[p.sa, p.sb])
        func.build(["d","sa"])
        print(func.run({"a": 2, "b": 4}))

        func.rebuild(["c"])
        print(func.run({"a": 2, "b": 4}))
    """

    DATA_BY = -1
    DATA_TO = 1
    FUNC_SUSP = 0
    FUNC_EXEC = 1

    def __init__(self):
        self.func_node = OrderedDict()
        self.data_node = OrderedDict()
        self.start = []
        self.need_variables = []
        self.cacu_data = set()

    def add_func_node(self, func, by, to, func_name=None):
        '''

        :param func:
        :param by:
        :param to:
        :param replace:
        :param args: 如果最后的需求变量中需要该函数的执行，但是该函数确实不想执行，那么该
        :param func_name:
        :return:
        '''
        assert hasattr(func, "__name__") or func_name is not None

        i = 0
        func_name = "{}_{}".format(getattr(func, "__name__", func_name), i)
        assert not func_name.startswith("__"), "function name must be not started with '__'"
        while func_name in self.func_node:
            func_name = "{}_{}".format(func_name, i)
            i += 1

        self.func_node[func_name] = dict(
            func=func,
            by=by,
            to=to,
            force_disable=False,
            replace=None,
            exec=False
        )
        return func_name

    def _add_data_node(self, name, data, is_by):
        self.data_node.setdefault(name, dict())
        self.data_node[name]["data"] = data
        self.data_node[name].setdefault("by", False)
        self.data_node[name].setdefault("to", False)
        self.data_node[name].setdefault("func", [])
        if is_by:
            self.data_node[name]["by"] = True
        else:
            self.data_node[name]["to"] = True

    def _bind_func(self, name, func_name):
        assert name in self.data_node, name
        self.data_node[name]["func"].append(func_name)

    def _update_data_node(self, name, data):
        self.data_node[name]["data"] = data

    def _enable_func(self, name):
        self.func_node[name]["exec"] = True
        for var_name in self.func_node[name]["by"]:
            self._label_cacu_data(var_name)
            for func_name in self.data_node[var_name]["func"]:
                self._enable_func(func_name)

    def _label_cacu_data(self, name):
        self.cacu_data.add(name)

    def _build_by_list(self, func_name):
        assert func_name in self.func_node
        res = []
        for k in self.func_node[func_name]["by"]:
            assert self.data_node[k]["by"], k
            res.append(self.data_node[k]["data"])
        return res

    def force_disable(self, func_flag, replace=None):
        self.func_node[func_flag]["force_disable"] = True
        self.func_node[func_flag]["replace"] = replace

    def build(self, need_variables: Iterable):
        for k, v in self.func_node.items():
            for var_name in v["by"]:
                self._add_data_node(var_name, v["replace"], is_by=True)
                self.start.append(var_name)
            for var_name in v["to"]:
                self._add_data_node(var_name, v["replace"], is_by=False)
                self._bind_func(var_name, k)

        self.need_variables = need_variables
        for var_name in need_variables:
            assert var_name in self.data_node, "variable '{}' not be assigned, please check the code."
            self._label_cacu_data(var_name)
            for func_name in self.data_node[var_name]["func"]:
                self._enable_func(func_name)

    def rebuild(self, need_variables: Iterable):
        self.need_variables.clear()
        self.data_node.clear()
        self.cacu_data.clear()
        for k, v in self.func_node.items():
            v["exec"] = False

        self.build(need_variables)

    def run(self, feed_dict: dict) -> VarDict:
        res = VarDict()

        for k, v in feed_dict.items():
            self._update_data_node(k, v)

        for k, v in self.func_node.items():
            if not v["exec"]:
                continue

            args = self._build_by_list(k)
            func_res = v["func"](*args)
            if not isinstance(func_res, Iterable):
                func_res = [func_res]

            for to_n, data in zip(v["to"], func_res):
                self._update_data_node(to_n, data)
                if to_n in self.need_variables:
                    res[to_n] = data
                else:
                    # elif to_n in self.cacu_data:
                    mid_res = res.setdefault("__mid_var_res", {})
                    mid_res[to_n] = data

        return res


class Branch():
    '''
    基于FuncGraph的更简单的使用类，用于更好的控制执行逻辑::

        def add(a, b):
            print("add")
            return a + b

        def multi(a, b):
            print("multi")
            return a * b

        def swap(a, b):
            print("swap")
            return b, a

        p = PseudoVar()
        br = Branch()

        br.add_node = add, (p.a, p.b), (p.c)
        br.multi = multi, (p.a, p.c), (p.d)
        br.swap = swap, (p.a, p.b), (p.sa, p.sb)

        res = br.run_for(p.d, p.sa, a=1, b=2)
        print(res,type(res))
        print(res.d)

        # or
        br.add_needs(p.d,p.sa)
        br.run(a=1,b=2)
    '''

    def __init__(self):
        self.init = True
        self.graph = FuncGraph()
        self.need_set = set()
        self.init = False

    def __setattr__(self, key, value):
        if (key == "init" and not hasattr(self, key)) or self.init:
            super().__setattr__(key, value)
            return

        assert 2 <= len(value) <= 5
        force_dis = False
        replace = None
        if len(value) == 2:
            func, by, to, force_dis, replace = value
        elif len(value) == 3:
            func, by, to = value
        elif len(value) == 4:
            func, by, to, force_dis = value
        else:
            func, by, to, force_dis, replace = value

        func_flag = self.graph.add_func_node(func, by, to, key)
        if force_dis:
            self.graph.force_disable(func_flag=func_flag, replace=replace)

    def add_needs(self, *args):
        for i in args:
            self.need_set.add(i)

    def clear_needs(self):
        self.need_set.clear()

    def build(self):
        self.graph.rebuild(self.need_set)

    def run(self, **kwargs):
        return self.graph.run(feed_dict=kwargs)

    def run_for(self, *args, **kwargs) -> VarDict:
        self.graph.rebuild(need_variables=args)
        return self.graph.run(feed_dict=kwargs)

    def __call__(self, by: list, to: list, force_dis: bool = False, replace=None):
        return by, to, force_dis, replace
