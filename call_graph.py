'''
build a graph similar to the following
nodes=methods
link[caller, callee] = (args)
variable_types[(var, class)] = type
variable_types[(var, method)] = type

'''
import os
import plyj.model as m
from collections import defaultdict

class Callgraph:
    #builds call graph linking functions
    #based on method invocations and declarations
    def __init__(self, tree):
        self.tree = tree
        # print "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
        # print tree        # Used for debugging
        # print "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
        self.graph = {'in': defaultdict(set)
                       , 'out': defaultdict(set)}
        self.nodes = []
        self._from_tree()

    def _from_tree(self):
        if not self.tree:
           return
        for type_decl in self.tree.type_declarations:
            if not hasattr(type_decl, 'body'):
                continue
            for decl in type_decl.body:
                if type(decl) is m.MethodDeclaration:
                    node = Node.from_obj(decl)
                    self.nodes.append(node)
                    for mm in node.links:
                        self.graph['out'][node.name].add(mm)
                        self.graph['in'][mm].add(node.name)

    @classmethod
    def from_file(cls, filename, parser):
        tree = parser.parse_file(
                    filename)
        #print "Received tree ", tree
        return Callgraph(tree)

class Node:
    #stores class information

    def __init__(self, 
           name, return_val, params, 
             methods, statement_types):
        self.name = name
        self.return_val = return_val
        self.params = params
        self.links = set()
        self.body = statement_types
        self._process_links(methods)

    def _process_links(self, methods):
        for md in methods:
            vs, ms = md
            for vv in vs:
                if type(vv) is tuple:
                    vtype, vname = vv
                else:
                    vname = vv
                    vtype = None
                self.params[vname] = vtype
            for mm in ms:
                if type(mm) is tuple:
                    self.links.add(mm[1])
                else:
                    self.links.add(mm)
                         

    @classmethod
    def from_obj(cls, method_decl): 
        name = method_decl.name
        return_val = cls.handle_return_type(
                       method_decl)
        params = cls.handle_params(method_decl)
        methods, statement_types = \
                cls.handle_body(method_decl)
        return Node(name, return_val,
                     params, methods, 
                       statement_types)

    @classmethod
    def handle_return_type(cls, method_decl):
        return_val = None
        if method_decl.return_type \
           is not None:
            return_val = obj_handler( 
                         method_decl.return_type)
        return return_val

    @classmethod
    def handle_params(cls, method_decl):
        params = {}
        for param in method_decl.parameters:
            param_type = obj_handler(param.type)
            if hasattr(param, 'name'):
                param_name = obj_handler(param.name)
                params[param_name] = param_type
            elif hasattr(param, 'variable'):
                param_name = obj_handler(param.variable)
                params[param_name] = param_type
            else:
                print("Unhandled param", param)
        return params

    @classmethod
    def handle_body(cls, method_decl):
        if method_decl.body is None:
            return [], []
        methods = [] #track function calls
        statement_types = []
        
        function_handling = {
            m.ClassInitializer : cls.cls_init,
            m.VariableDeclaration : \
                   cls.var_decl,
            m.Conditional: \
                   cls.condition_decl,
            m.Block: \
                   cls.block_decl,
            m.ArrayInitializer: \
                   cls.array_decl,
            m.MethodInvocation: \
                   cls.method_invoc,
            m.IfThenElse : \
                   cls.cond_decl,
            m.While: \
                   cls.loop_decl,
            m.For: \
                   cls.loop_decl,
            m.ForEach : \
                   cls.loop_decl,
            m.Switch: \
                   cls.switch_decl,
            m.SwitchCase: \
                   cls.switch_decl,
            m.DoWhile: \
                   cls.loop_decl,
            m.Try: \
                cls.try_decl,
            m.Catch: \
                cls.try_decl,
            m.ConstructorInvocation: \
                cls.const_invoc,
            m.InstanceCreation : \
                cls.inst_invoc,
            m.ExpressionStatement: \
                cls.expr_decl,
            m.Assignment: \
                cls.assign_decl,
            m.Return:
                cls.return_decl
            }       
            
        for statement in method_decl.body:
            for kk, vv in function_handling.items():
                if type(statement) is kk:
                    statement_types.append(
                        statement)
                    methods += \
                       vv(statement, \
                          function_handling)
        return methods, statement_types

    @classmethod
    def cls_init(cls, s, f):
        methods = []
        if not hasattr(s, 'block'):
            return methods
        if s.block is None:
            return methods

        if type(s.block) is m.Block:
            return f[m.Block](s.block, f)
        elif type(s.block) is list:
            for bstatement in s.block:
                for kk, vv in f.items():
                    if type(bstatement) \
                         is kk:
                        methods += vv(bstatement, f)
        return methods
    
    @classmethod
    def var_decl(cls, s, f):
        methods = []
        vtype = None
        vtype = obj_handler(s.type)
        vname = None
        vs = []
        ms = []
        for vv in s.variable_declarators:
            if type(vv) \
                is m.VariableDeclarator:
                vname = obj_handler(vv.variable)
                vs.append((vtype, vname))
            if type(vv.initializer) \
                 is m.MethodInvocation:
                ms.append((type(vv.initializer),
                         obj_handler(
                       vv.initializer)))
            elif type(vv.initializer) \
                  is m.InstanceCreation:
                ms.append((type(vv.initializer),
                         obj_handler(
                       vv.initializer.type)))
        methods.append((vs, ms))
        return methods
                  
    def condition_decl(cls, s, f):
        print("Conditional", s)
        return []
    
    @classmethod
    def cast_decl(cls, s, f):
        methods = []
        if not hasattr(s, 'target'):
           return methods
        if s.target is None:
           return methods
        for kk, vv in f.items():
            if type(s.target) is kk:
                methods = vv(s.target, f)
        return methods

    @classmethod
    def block_decl(cls, s, f):
        methods = []
        for bstatement in s.statements:
            for kk, vv in f.items():
                if type(bstatement) is kk:
                    methods += vv(bstatement, f)
        return methods
    @classmethod
    def array_decl(cls, s, f):
        print("Array", s)
        return []

    @classmethod
    def method_invoc(cls, s, f):
        methods = []
        vs = []
        ms = []
        ms.append(obj_handler(s))
        if hasattr(s, 'target'):
            if s.target is not None:
                vs.append((type(s.target),
                   obj_handler(s.target)))
        methods.append((vs, ms))
        return methods
                          
    @classmethod
    def cond_decl(cls, s, f):
        methods = []
        for kk, vv in f.items():
            if type(s.if_true) is \
                 kk:
                methods += \
                  vv(s.if_true, f)
            if type(s.if_false) is \
                 kk:
                methods += \
                  vv(s.if_false, f)
        return methods
    
    @classmethod
    def loop_decl(cls, s, f):
        methods = []
        if s.body is None:
            return methods
        body = s.body
        if not type(s.body) is list:
            body = [s.body]
        for ss in body:
            for kk, vv in f.items():
                if type(ss) is kk:
                    methods += vv(ss, f)
        return methods

    @classmethod
    def try_decl(cls, s, f):
        #print(s)
        methods = []
        if s.block is None:
            return methods

        for kk, vv in f.items():
           if type(s.block) is kk:
              methods += vv(s.block, f)
        return methods

    @classmethod
    def switch_decl(cls, s, f):
        #print("Switch", s)
        return []

    @classmethod
    def const_invoc(cls, s, f):
        print("Constructor", s)
        return []

    @classmethod
    def inst_invoc(cls, s, f):
        methods = []
        vs = []
        ms = []
        ms.append(obj_handler(s.type))
        methods.append((vs, ms))
        for kk, vv in f.items():
            for ss in s.body:
                if type(ss) is kk:
                     methods += \
                       vv(ss, f)
        return methods
    @classmethod
    def expr_decl(cls, s, f):
        methods = []
        for kk, vv in f.items():
            if type(s.expression) is kk:
                methods = vv(s.expression, f)
        return methods
    
    @classmethod
    def assign_decl(cls, s, f):
        methods = []
        ms = []
        vs = []
        if s.lhs is not None:
            vs.append((None, 
               obj_handler(s.lhs)))
        if s.rhs is not None:
            if type(s.rhs) \
                 is m.MethodInvocation:
                ms.append((type(s.rhs),
                         obj_handler(
                       s.rhs)))
            elif type(s.rhs) \
                  is m.InstanceCreation:
                 ms.append((type(s.rhs),
                         obj_handler(
                       s.rhs.type)))
        methods.append((vs, ms))
        return methods

    @classmethod
    def return_decl(cls, s, f):
        methods = []
        ms = []
        vs = []
        stype = s.result
        if type(stype) is \
           m.MethodInvocation:
            ms.append((type(stype),
                   obj_handler(
                   stype)))
        elif type(stype) \
              is m.InstanceCreation:
            ms.append((type(stype),
                         obj_handler(
                       stype.type)))
        methods.append((vs, ms))
        return methods

def obj_handler(obj):
    if type(obj) is str:
        return obj
    else:
        if hasattr(obj, 'name'):
            if type(obj.name) is str:
                return obj.name
            elif type(obj.name) is m.Name:
                return obj.name.value
            elif type(obj.name) is m.Type:
                return obj.name.name.value
            else:
                print('Unknown obj handler', obj)
        else:
            if hasattr(obj, 'value'):
                if type(obj.value) is str:
                    return obj.value
                else:
                    return None
