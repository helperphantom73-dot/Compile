#!/usr/bin/env python3
"""Axon Compiler v7.0 - Zero-Copy mmap + Brain-Like Cortical Routing + RTL + GitHub CI.
Run: python3 axon_bundle.py
Then: python3 -m axon.cli selfcheck
"""
import os, sys, subprocess, shutil

FILES = {}

FILES["pyproject.toml"] = r'''[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "axon-compiler"
version = "7.0.0"
description = "Axon: zero-copy, brain-routed tensor DSL compiler (LLVM + C runtime + RTL)"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["llvmlite>=0.40,<0.45"]
[project.scripts]
axon = "axon.cli:main"
[tool.setuptools.packages.find]
where = ["."]
include = ["axon*"]
[tool.setuptools.package-data]
"axon.runtime" = ["*.c", "*.h", "Makefile"]
'''

FILES["README.md"] = r'''# Axon v7.0
Auditable Tensor DSL -> LLVM object -> Zero-Copy `.bin` + `.axn3` mmap / static library / RTL core.

## Verify
python3 axon_bundle.py
python3 -m axon.cli selfcheck
make sanitize

## Capabilities & Honest Limits
- **Brain-Like Cortical Routing (`--brain` / `neuro_matmul`)**: Keeps compact region signatures (`f32[K, R]`) hot in RAM and activates/pages only the top 50% weight regions per input.
- **Zero-Copy `mmap` (`--external-weights`)**: Streams `<output>.axn3` via POSIX `mmap` + `MADV_RANDOM` with zero heap copy.
- **O(log N) Column-Indexed FP4 Residuals** + **4-row SIMD micro-kernel** (Dense F32, FP4, Ternary).
- Limits: f32 DSL, rank 1..4, one function per file, 2-D matmul, static shapes; `--emit-silicon` emits one Q8.8 MAC core + weight hex + `$readmemh` testbench.
'''

FILES[".github/workflows/ci.yml"] = r'''name: Axon v7.0 CI
on: [push, pull_request]
jobs:
  build-and-verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install system deps & llvmlite
        run: |
          sudo apt-get update && sudo apt-get install -y build-essential iverilog
          python3 -m pip install --upgrade pip "llvmlite>=0.40,<0.45"
      - name: Unpack & build
        run: python3 axon_bundle.py
      - name: Selfcheck (Unit Tests + Verilog RTL Sim)
        run: python3 -m axon.cli selfcheck
      - name: Sanitizer Suite (ASan + UBSan)
        run: make sanitize
      - name: Upload compiled artifacts
        uses: actions/upload-artifact@v4
        with:
          name: axon-v7-linux-x86_64
          path: |
            transformer.bin
            transformer_brain.bin
            transformer_brain.bin.axn3
            axon_chip.v
'''

FILES["Makefile"] = r'''PYTHON ?= python3
CC ?= cc
.PHONY: all runtime test selfcheck sanitize clean
all: runtime
runtime:
@@$(PYTHON) -c "from axon.runtime_paths import ensure_runtime; ensure_runtime(verbose=True)"
test: runtime
@@$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v
selfcheck: runtime
@@$(PYTHON) -m axon.cli selfcheck
sanitize:
@@AXON_SANITIZE=1 $(PYTHON) -c "from axon.runtime_paths import ensure_runtime; ensure_runtime(verbose=True)"
@@$(CC) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -pthread -Iaxon/runtime tests/test_c_runtime.c axon/runtime/axon_runtime.c -lm -o /tmp/axon_c_san
@@/tmp/axon_c_san && rm -f /tmp/axon_c_san
@@AXON_SANITIZE=1 $(PYTHON) -m unittest tests.test_axon.TestEndToEnd -v
clean:
@@$(MAKE) -C axon/runtime clean
@@rm -rf build dist *.egg-info
@@find . -type d -name __pycache__ -exec rm -rf {} +
'''

FILES["axon/__init__.py"] = r'''__version__ = "7.0.0"
'''

FILES["axon/errors.py"] = r'''class AxonError(Exception):
    def __init__(self, msg, line=0, col=0, filename="<axon>"):
        self.msg, self.line, self.col, self.filename = msg, line, col, filename
        super().__init__(msg)
    def __str__(self):
        return f"{self.filename}:{self.line}:{self.col}: error: {self.msg}" if self.line else f"{self.filename}: error: {self.msg}"
class LexError(AxonError): pass
class ParseError(AxonError): pass
class SemanticError(AxonError): pass
class CodegenError(AxonError): pass
'''

FILES["axon/frontend.py"] = r'''import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
from .errors import LexError, ParseError, SemanticError

class TK(Enum):
    IDENT=auto(); INT=auto(); FLOAT=auto(); FN=auto(); LET=auto(); RETURN=auto(); AS=auto()
    LP=auto(); RP=auto(); LB=auto(); RB=auto(); LC=auto(); RC=auto()
    COMMA=auto(); COLON=auto(); ARROW=auto(); ASSIGN=auto(); SEMI=auto(); PLUS=auto(); MINUS=auto(); STAR=auto(); EOF=auto()

@dataclass(frozen=True)
class Tok: kind: TK; text: str; line: int; col: int
KW={"fn":TK.FN,"let":TK.LET,"return":TK.RETURN,"as":TK.AS}
SINGLE={"(":TK.LP,")":TK.RP,"[":TK.LB,"]":TK.RB,"{":TK.LC,"}":TK.RC,",":TK.COMMA,":":TK.COLON,";":TK.SEMI,"=":TK.ASSIGN,"+":TK.PLUS,"-":TK.MINUS,"*":TK.STAR}
NUM=re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"); ID=re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

class Lexer:
    def __init__(self, src, filename="<axon>"): self.s=src; self.f=filename; self.i=0; self.line=1; self.col=1
    def _adv(self,n):
        for _ in range(n):
            ch=self.s[self.i]; self.i+=1
            if ch=="\n": self.line+=1; self.col=1
            else: self.col+=1
    def tokenize(self):
        toks=[]; n=len(self.s)
        while self.i<n:
            ch=self.s[self.i]
            if ch in " \t\r\n\f\v": self._adv(1); continue
            if ch=="#" or self.s.startswith("//",self.i):
                while self.i<n and self.s[self.i]!="\n": self._adv(1)
                continue
            l,c=self.line,self.col
            if self.s.startswith("->",self.i): self._adv(2); toks.append(Tok(TK.ARROW,"->",l,c)); continue
            if ch in SINGLE: self._adv(1); toks.append(Tok(SINGLE[ch],ch,l,c)); continue
            m=ID.match(self.s,self.i)
            if m: t=m.group(0); self._adv(len(t)); toks.append(Tok(KW.get(t,TK.IDENT),t,l,c)); continue
            m=NUM.match(self.s,self.i)
            if m: t=m.group(0); self._adv(len(t)); toks.append(Tok(TK.FLOAT if ("." in t or "e" in t or "E" in t) else TK.INT,t,l,c)); continue
            raise LexError(f"unexpected character {ch!r}",l,c,self.f)
        toks.append(Tok(TK.EOF,"",self.line,self.col)); return toks

@dataclass(frozen=True)
class Loc: line:int=1; col:int=1
class Node: loc:Loc
@dataclass
class TypeExpr(Node): name:str; shape:List[int]; loc:Loc=field(default_factory=Loc)
@dataclass
class Param(Node): name:str; type_expr:TypeExpr; loc:Loc=field(default_factory=Loc)
@dataclass
class TensorLit(Node): values:List[float]; shape:List[int]; dtype:str; loc:Loc=field(default_factory=Loc)
@dataclass
class VarRef(Node): name:str; loc:Loc=field(default_factory=Loc)
@dataclass
class Call(Node): callee:str; args:List[Node]; loc:Loc=field(default_factory=Loc); shape:Optional[List[int]]=None; dtype:Optional[str]=None
@dataclass
class Binary(Node): op:str; left:Node; right:Node; loc:Loc=field(default_factory=Loc); shape:Optional[List[int]]=None; dtype:Optional[str]=None
@dataclass
class Let(Node): name:str; type_expr:Optional[TypeExpr]; value:Node; loc:Loc=field(default_factory=Loc)
@dataclass
class Return(Node): value:Node; loc:Loc=field(default_factory=Loc)
@dataclass
class ExprStmt(Node): expr:Node; loc:Loc=field(default_factory=Loc)
@dataclass
class Function(Node): name:str; params:List[Param]; ret:TypeExpr; body:List[Node]; loc:Loc=field(default_factory=Loc)
@dataclass
class Program(Node): functions:List[Function]; loc:Loc=field(default_factory=Loc)

class Parser:
    def __init__(self, toks, filename="<axon>"): self.t=toks; self.f=filename; self.p=0
    def _peek(self): return self.t[self.p]
    def _check(self,k): return self._peek().kind==k
    def _adv(self):
        x=self._peek()
        if x.kind!=TK.EOF: self.p+=1
        return x
    def _expect(self,k,msg):
        if self._check(k): return self._adv()
        c=self._peek(); raise ParseError(f"{msg}, got {c.kind.name} ({c.text!r})",c.line,c.col,self.f)
    def _err(self,msg,tok=None): t=tok or self._peek(); raise ParseError(msg,t.line,t.col,self.f)
    def parse(self):
        fns=[]
        while not self._check(TK.EOF): fns.append(self._function())
        if not fns: self._err("empty program")
        if len(fns)>1: self._err("only one function per module is supported")
        return Program(fns)
    def _function(self):
        ft=self._expect(TK.FN,"expected 'fn'"); nt=self._expect(TK.IDENT,"expected name"); self._expect(TK.LP,"expected '('"); params=[]
        if not self._check(TK.RP):
            while True:
                pn=self._expect(TK.IDENT,"expected param"); self._expect(TK.COLON,"expected ':'")
                params.append(Param(pn.text,self._type(),Loc(pn.line,pn.col)))
                if self._check(TK.COMMA):
                    self._adv()
                    if self._check(TK.RP): self._err("trailing comma")
                    continue
                break
        self._expect(TK.RP,"expected ')'"); self._expect(TK.ARROW,"expected '->'"); rt=self._type(); self._expect(TK.LC,"expected '{'"); body=[]
        while not self._check(TK.RC):
            if self._check(TK.EOF): self._err("unterminated body")
            body.append(self._stmt())
        self._expect(TK.RC,"expected '}'")
        return Function(nt.text,params,rt,body,Loc(ft.line,ft.col))
    def _type(self):
        dt=self._expect(TK.IDENT,"expected dtype"); sh=[]
        if self._check(TK.LB):
            self._adv()
            if self._check(TK.RB): self._err("empty shape")
            while True:
                d=self._expect(TK.INT,"expected int dim"); sh.append(int(d.text))
                if self._check(TK.COMMA):
                    self._adv()
                    if self._check(TK.RB): self._err("trailing comma")
                    continue
                break
            self._expect(TK.RB,"expected ']'")
        return TypeExpr(dt.text,sh,Loc(dt.line,dt.col))
    def _stmt(self):
        if self._check(TK.LET):
            t=self._adv(); n=self._expect(TK.IDENT,"expected name"); te=None
            if self._check(TK.COLON): self._adv(); te=self._type()
            self._expect(TK.ASSIGN,"expected '='"); v=self._expr(); self._expect(TK.SEMI,"expected ';'")
            return Let(n.text,te,v,Loc(t.line,t.col))
        if self._check(TK.RETURN):
            t=self._adv(); v=self._expr(); self._expect(TK.SEMI,"expected ';'")
            return Return(v,Loc(t.line,t.col))
        e=self._expr(); self._expect(TK.SEMI,"expected ';'"); return ExprStmt(e,e.loc)
    def _expr(self):
        n=self._term()
        while self._check(TK.PLUS) or self._check(TK.MINUS):
            op=self._adv(); n=Binary("+" if op.kind==TK.PLUS else "-",n,self._term(),Loc(op.line,op.col))
        return n
    def _term(self):
        n=self._primary()
        while self._check(TK.STAR):
            op=self._adv(); n=Binary("*",n,self._primary(),Loc(op.line,op.col))
        return n
    def _primary(self):
        t=self._peek()
        if t.kind==TK.LB: return self._tensor()
        if t.kind==TK.LP: self._adv(); e=self._expr(); self._expect(TK.RP,"expected ')'"); return e
        if t.kind==TK.IDENT:
            self._adv(); loc=Loc(t.line,t.col)
            if self._check(TK.LP):
                self._adv(); args=[]
                if not self._check(TK.RP):
                    while True:
                        args.append(self._expr())
                        if self._check(TK.COMMA):
                            self._adv()
                            if self._check(TK.RP): self._err("trailing comma")
                            continue
                        break
                self._expect(TK.RP,"expected ')'"); return Call(t.text,args,loc)
            return VarRef(t.text,loc)
        if t.kind in (TK.INT,TK.FLOAT): self._err("bare numbers only inside '[...]'")
        if t.kind==TK.MINUS: self._err("unary '-' only inside tensor literal")
        self._err(f"unexpected {t.kind.name}")
    def _tensor(self):
        lb=self._adv(); vals=[]
        if not self._check(TK.RB):
            while True:
                sign=1.0
                if self._check(TK.MINUS): self._adv(); sign=-1.0
                elif self._check(TK.PLUS): self._adv()
                num=self._adv()
                if num.kind not in (TK.INT,TK.FLOAT): self._err("expected number",num)
                v=sign*float(num.text)
                if v!=v or v in (float("inf"),float("-inf")): self._err("non-finite literal",num)
                vals.append(v)
                if self._check(TK.COMMA):
                    self._adv()
                    if self._check(TK.RB): self._err("trailing comma")
                    continue
                break
        self._expect(TK.RB,"expected ']'"); dt,sh="f32",[len(vals)]
        if self._check(TK.AS):
            self._adv(); tx=self._type(); dt=tx.name
            if tx.shape: sh=tx.shape
        return TensorLit(vals,sh,dt,Loc(lb.line,lb.col))

MAX_DIMS=4; DTYPES={"f32"}
ARITY={"matmul":2,"neuro_matmul":3,"add":2,"sub":2,"mul":2,"relu":1,"gelu":1,"silu":1,"sigmoid":1,"softmax":1,"transpose":1,"layernorm":3,"equilibrium":3}
@dataclass
class TypeInfo: dtype:str; shape:List[int]

class Semantic:
    def __init__(self, filename="<axon>"): self.f=filename; self.sym={}
    def _err(self,msg,loc): raise SemanticError(msg,loc.line,loc.col,self.f)
    def _shape(self,sh,loc,what):
        if not (1<=len(sh)<=MAX_DIMS): self._err(f"{what}: rank {len(sh)} out of 1..{MAX_DIMS}",loc)
        if any(d<=0 for d in sh): self._err(f"{what}: dims must be positive {sh}",loc)
    def _dtype(self,dt,loc,what):
        if dt not in DTYPES: self._err(f"{what}: unsupported dtype {dt!r}",loc)
    @staticmethod
    def _numel(sh):
        n=1
        for d in sh: n*=d
        return n
    def _bcast(self,a,b,loc):
        n=max(len(a),len(b)); pa=[1]*(n-len(a))+list(a); pb=[1]*(n-len(b))+list(b); out=[]
        for x,y in zip(pa,pb):
            if x==y: out.append(x)
            elif x==1: out.append(y)
            elif y==1: out.append(x)
            else: self._err(f"cannot broadcast {a} with {b}",loc)
        self._shape(out,loc,"broadcast"); return out
    def analyze(self,prog):
        for fn in prog.functions: self._fn(fn)
        return prog
    def _fn(self,fn):
        self.sym.clear()
        if not fn.params: self._err("need at least one input",fn.loc)
        self._dtype(fn.ret.name,fn.ret.loc,"return"); self._shape(fn.ret.shape,fn.ret.loc,"return")
        for p in fn.params:
            if p.name in self.sym: self._err(f"duplicate param {p.name}",p.loc)
            self._dtype(p.type_expr.name,p.type_expr.loc,"param"); self._shape(p.type_expr.shape,p.type_expr.loc,"param")
            self.sym[p.name]=TypeInfo(p.type_expr.name,list(p.type_expr.shape))
        rets=[i for i,s in enumerate(fn.body) if isinstance(s,Return)]
        if len(rets)!=1: self._err("exactly one return required",fn.loc)
        if rets[0]!=len(fn.body)-1: self._err("return must be last",fn.body[rets[0]].loc)
        for s in fn.body:
            if isinstance(s,Let):
                if s.name in self.sym: self._err(f"redefinition of {s.name} (single-assignment)",s.loc)
                inf=self._infer(s.value)
                if s.type_expr:
                    self._dtype(s.type_expr.name,s.type_expr.loc,"annotation")
                    if s.type_expr.name!=inf.dtype: self._err(f"dtype mismatch on {s.name}",s.loc)
                    if s.type_expr.shape and s.type_expr.shape!=inf.shape: self._err(f"shape mismatch on {s.name}",s.loc)
                self.sym[s.name]=inf
            elif isinstance(s,Return):
                r=self._infer(s.value)
                if r.dtype!=fn.ret.name: self._err("return dtype mismatch",s.loc)
                if r.shape!=fn.ret.shape: self._err(f"return shape {r.shape} != {fn.ret.shape}",s.loc)
            elif isinstance(s,ExprStmt): self._infer(s.expr)
    def _infer(self,e):
        if isinstance(e,TensorLit):
            self._dtype(e.dtype,e.loc,"literal"); self._shape(e.shape,e.loc,"literal")
            if self._numel(e.shape)!=len(e.values): self._err(f"literal has {len(e.values)} values, shape {e.shape} needs {self._numel(e.shape)}",e.loc)
            return TypeInfo(e.dtype,list(e.shape))
        if isinstance(e,VarRef):
            if e.name not in self.sym: self._err(f"undefined {e.name}",e.loc)
            t=self.sym[e.name]; return TypeInfo(t.dtype,list(t.shape))
        if isinstance(e,Binary):
            l,r=self._infer(e.left),self._infer(e.right)
            if l.dtype!=r.dtype: self._err("dtype mismatch",e.loc)
            sh=self._bcast(l.shape,r.shape,e.loc); e.shape,e.dtype=sh,l.dtype; return TypeInfo(l.dtype,sh)
        if isinstance(e,Call): return self._call(e)
        self._err("unsupported expression",getattr(e,"loc",Loc()))
    def _call(self,e):
        c=e.callee
        if c not in ARITY: self._err(f"unknown builtin {c!r}",e.loc)
        if len(e.args)!=ARITY[c]: self._err(f"{c} expects {ARITY[c]} args",e.loc)
        a=[self._infer(x) for x in e.args]
        if any(x.dtype!=a[0].dtype for x in a): self._err(f"{c}: dtype mismatch",e.loc)
        dt=a[0].dtype
        if c=="matmul":
            x,y=a
            if len(x.shape)!=2 or len(y.shape)!=2: self._err("matmul requires 2-D",e.loc)
            if x.shape[1]!=y.shape[0]: self._err(f"matmul inner dims {x.shape} x {y.shape}",e.loc)
            sh=[x.shape[0],y.shape[1]]
        elif c=="neuro_matmul":
            x,y,s=a
            if len(x.shape)!=2 or len(y.shape)!=2 or len(s.shape)!=2: self._err("neuro_matmul requires 2-D",e.loc)
            if x.shape[1]!=y.shape[0] or s.shape[0]!=y.shape[0] or not (1<=s.shape[1]<=y.shape[1]): self._err("neuro_matmul shape mismatch",e.loc)
            sh=[x.shape[0],y.shape[1]]
        elif c in ("add","sub","mul"): sh=self._bcast(a[0].shape,a[1].shape,e.loc)
        elif c=="transpose":
            if len(a[0].shape)!=2: self._err("transpose requires 2-D",e.loc)
            sh=[a[0].shape[1],a[0].shape[0]]
        elif c in ("relu","gelu","silu","sigmoid","softmax"): sh=list(a[0].shape)
        elif c in ("layernorm","equilibrium"):
            d=a[0].shape[-1]
            if c=="layernorm":
                if a[1].shape!=[d] or a[2].shape!=[d]: self._err("layernorm params mismatch",e.loc)
            else:
                if a[1].shape!=[d,d] or a[2].shape!=[d]: self._err("equilibrium params mismatch",e.loc)
            sh=list(a[0].shape)
        else: self._err("internal",e.loc); sh=[]
        self._shape(sh,e.loc,c); e.shape,e.dtype=sh,dt; return TypeInfo(dt,sh)
'''

FILES["axon/ir.py"] = r'''from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from .errors import CodegenError
from .frontend import Let, Return, ExprStmt, Call, Binary, TensorLit, VarRef

class OpKind(Enum):
    INPUT=auto(); CONSTANT=auto(); MATMUL=auto(); NEURO_MATMUL=auto()
    ADD=auto(); SUB=auto(); MUL=auto()
    RELU=auto(); GELU=auto(); SILU=auto(); SIGMOID=auto()
    SOFTMAX=auto(); LAYERNORM=auto(); TRANSPOSE=auto(); EQUILIBRIUM=auto()
    FUSED_MATMUL_RELU=auto(); FUSED_MATMUL_GELU=auto(); FUSED_MATMUL_SILU=auto()
    FUSED_MATMUL_BIAS=auto()
    FUSED_MATMUL_BIAS_RELU=auto(); FUSED_MATMUL_BIAS_GELU=auto(); FUSED_MATMUL_BIAS_SILU=auto()

MATMUL_LIKE=(OpKind.MATMUL,OpKind.NEURO_MATMUL,OpKind.FUSED_MATMUL_RELU,OpKind.FUSED_MATMUL_GELU,OpKind.FUSED_MATMUL_SILU,
             OpKind.FUSED_MATMUL_BIAS,OpKind.FUSED_MATMUL_BIAS_RELU,OpKind.FUSED_MATMUL_BIAS_GELU,OpKind.FUSED_MATMUL_BIAS_SILU)

@dataclass
class WeightBlob:
    name:str; data:List[float]; shape:List[int]; dtype:str="f32"; quant:str="none"
    packed:Optional[bytes]=None; scales:Optional[bytes]=None; scale_f:float=1.0
    residuals:List[Tuple[int,float]]=field(default_factory=list)
    def numel(self):
        n=1
        for d in self.shape: n*=d
        return n

@dataclass
class IRNode:
    id:int; op:OpKind; inputs:List[int]; attrs:Dict[str,Any]; shape:Tuple[int,...]; dtype:str; name:str=""

@dataclass
class Graph:
    nodes:List[IRNode]=field(default_factory=list); inputs:List[int]=field(default_factory=list)
    outputs:List[int]=field(default_factory=list); weights:Dict[str,WeightBlob]=field(default_factory=dict)
    _idx:Dict[int,IRNode]=field(default_factory=dict,repr=False)
    def reindex(self): self._idx={n.id:n for n in self.nodes}
    def get(self,nid):
        if not self._idx: self.reindex()
        return self._idx.get(nid)
    def fresh_id(self): return max((n.id for n in self.nodes),default=-1)+1
    def topo(self):
        self.reindex(); indeg={n.id:0 for n in self.nodes}; adj={n.id:[] for n in self.nodes}
        for n in self.nodes:
            for i in n.inputs:
                if i not in indeg: raise CodegenError(f"dangling input %{i}")
                adj[i].append(n.id); indeg[n.id]+=1
        ready=sorted(k for k,v in indeg.items() if v==0); out=[]
        while ready:
            c=ready.pop(0); out.append(self._idx[c]); new=[]
            for nb in adj[c]:
                indeg[nb]-=1
                if indeg[nb]==0: new.append(nb)
            if new: ready=sorted(ready+new)
        if len(out)!=len(self.nodes): raise CodegenError("cycle in graph")
        return out
    def replace(self,old,new):
        oids={n.id for n in old}; sink=old[-1].id
        for n in self.nodes: n.inputs=[new.id if i==sink else i for i in n.inputs]
        self.outputs=[new.id if i==sink else i for i in self.outputs]
        self.nodes=[n for n in self.nodes if n.id not in oids]+[new]; self.reindex()
    def alias_weights(self): return {n.name:self.weights[n.attrs["weight"]] for n in self.nodes if n.op==OpKind.CONSTANT}

CALL_MAP={"matmul":OpKind.MATMUL,"neuro_matmul":OpKind.NEURO_MATMUL,"add":OpKind.ADD,"sub":OpKind.SUB,"mul":OpKind.MUL,
          "relu":OpKind.RELU,"gelu":OpKind.GELU,"silu":OpKind.SILU,"sigmoid":OpKind.SIGMOID,
          "softmax":OpKind.SOFTMAX,"layernorm":OpKind.LAYERNORM,"transpose":OpKind.TRANSPOSE,"equilibrium":OpKind.EQUILIBRIUM}
BIN_MAP={"+":OpKind.ADD,"-":OpKind.SUB,"*":OpKind.MUL}

class IRBuilder:
    def __init__(self): self.g=Graph(); self.nid=0; self.vmap={}; self._cc={}
    def _fresh(self): i=self.nid; self.nid+=1; return i
    def build(self,prog):
        for fn in prog.functions:
            for p in fn.params:
                i=self._fresh()
                self.g.nodes.append(IRNode(i,OpKind.INPUT,[],{"param":p.name},tuple(p.type_expr.shape),p.type_expr.name,p.name))
                self.g.inputs.append(i); self.vmap[p.name]=i
            for s in fn.body:
                if isinstance(s,Let): self.vmap[s.name]=self._expr(s.value,s.name)
                elif isinstance(s,Return): self.g.outputs.append(self._expr(s.value,"result"))
                elif isinstance(s,ExprStmt): self._expr(s.expr)
        self.g.reindex(); return self.g
    def _expr(self,e,name=""):
        if isinstance(e,VarRef): return self.vmap[e.name]
        if isinstance(e,TensorLit):
            key=(tuple(e.shape),tuple(e.values)); wn=self._cc.get(key); i=self._fresh()
            if wn is None: wn=f"w{i}"; self._cc[key]=wn; self.g.weights[wn]=WeightBlob(wn,list(e.values),list(e.shape),e.dtype)
            self.g.nodes.append(IRNode(i,OpKind.CONSTANT,[],{"weight":wn},tuple(e.shape),e.dtype,name or wn)); return i
        if isinstance(e,Binary):
            l,r=self._expr(e.left),self._expr(e.right); i=self._fresh()
            self.g.nodes.append(IRNode(i,BIN_MAP[e.op],[l,r],{},tuple(e.shape or ()),e.dtype or "f32",name)); return i
        if isinstance(e,Call):
            args=[self._expr(a) for a in e.args]; i=self._fresh()
            self.g.nodes.append(IRNode(i,CALL_MAP[e.callee],args,{},tuple(e.shape or ()),e.dtype or "f32",name)); return i
        raise CodegenError(f"unsupported AST {type(e).__name__}")
'''

FILES["axon/passes.py"] = r'''import math
from .ir import OpKind, IRNode, WeightBlob, MATMUL_LIKE

class DCE:
    def run(self,g):
        live=set(g.outputs); stack=list(g.outputs)
        while stack:
            n=g.get(stack.pop())
            if not n: continue
            for i in n.inputs:
                if i not in live: live.add(i); stack.append(i)
        live.update(g.inputs)
        g.nodes=[n for n in g.nodes if n.id in live]; g.reindex()
        used={n.attrs["weight"] for n in g.nodes if n.op==OpKind.CONSTANT}
        g.weights={k:v for k,v in g.weights.items() if k in used}; return g

class BrainRouterPass:
    def run(self,g):
        for n in list(g.topo()):
            if n.op!=OpKind.MATMUL: continue
            wn=g.get(n.inputs[1])
            if not wn or wn.op!=OpKind.CONSTANT or len(wn.shape)!=2: continue
            K,N=wn.shape
            if N<4: continue
            wb=g.weights[wn.attrs["weight"]]
            if wb.quant!="none" or wb.dtype!="f32": continue
            R=min(N,8) if N<=64 else min(64,max(4,N//32))
            sig=[0.0]*(K*R)
            for r in range(R):
                c0=(r*N)//R; c1=((r+1)*N)//R; inv=1.0/max(1,c1-c0)
                for k in range(K): sig[k*R+r]=sum(wb.data[k*N+c] for c in range(c0,c1))*inv
            sid=g.fresh_id(); sname=f"sig_{wb.name}_{sid}"
            g.weights[sname]=WeightBlob(sname,sig,[K,R],"f32")
            g.nodes.append(IRNode(sid,OpKind.CONSTANT,[],{"weight":sname},(K,R),"f32",sname))
            n.op=OpKind.NEURO_MATMUL; n.inputs=[n.inputs[0],n.inputs[1],sid]
        g.reindex(); return g

ACT={OpKind.RELU:"RELU",OpKind.GELU:"GELU",OpKind.SILU:"SILU"}
class Fusion:
    def run(self,g):
        changed=True
        while changed:
            changed=False
            for n in g.topo():
                if n.op in ACT and self._act(n,g): changed=True; break
                if n.op==OpKind.ADD and self._bias(n,g): changed=True; break
        return g
    def _uses(self,nid,g): return sum(n.inputs.count(nid) for n in g.nodes)+g.outputs.count(nid)
    def _bias_for(self,add,mm,g):
        others=[i for i in add.inputs if i!=mm.id]
        if not others: return None
        b=g.get(others[0])
        return b.id if b and len(b.shape)==1 and len(mm.shape)==2 and b.shape[0]==mm.shape[1] else None
    def _bias(self,node,g):
        for pid in node.inputs:
            mm=g.get(pid)
            if not mm or mm.op!=OpKind.MATMUL or self._uses(mm.id,g)!=1: continue
            b=self._bias_for(node,mm,g)
            if b is None: continue
            g.replace([mm,node],IRNode(node.id,OpKind.FUSED_MATMUL_BIAS,[mm.inputs[0],mm.inputs[1],b],{},node.shape,node.dtype,node.name)); return True
        return False
    def _act(self,node,g):
        if len(node.inputs)!=1: return False
        p=g.get(node.inputs[0])
        if not p or self._uses(p.id,g)!=1: return False
        tag=ACT[node.op]
        if p.op==OpKind.FUSED_MATMUL_BIAS:
            g.replace([p,node],IRNode(node.id,OpKind[f"FUSED_MATMUL_BIAS_{tag}"],list(p.inputs),{},node.shape,node.dtype,node.name)); return True
        if p.op==OpKind.MATMUL:
            g.replace([p,node],IRNode(node.id,OpKind[f"FUSED_MATMUL_{tag}"],list(p.inputs),{},node.shape,node.dtype,node.name)); return True
        if p.op==OpKind.ADD:
            for pid in p.inputs:
                mm=g.get(pid)
                if not mm or mm.op!=OpKind.MATMUL or self._uses(mm.id,g)!=1: continue
                b=self._bias_for(p,mm,g)
                if b is None: continue
                g.replace([mm,p,node],IRNode(node.id,OpKind[f"FUSED_MATMUL_BIAS_{tag}"],[mm.inputs[0],mm.inputs[1],b],{},node.shape,node.dtype,node.name)); return True
        return False

FP4_LEVELS=(0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0); FP4_ENC={v:i for i,v in enumerate(FP4_LEVELS)}
def _rha(x): return int(math.floor(x+0.5)) if x>=0 else -int(math.floor(-x+0.5))
def round_fp4(v):
    s=-1.0 if math.copysign(1.0,v)<0 else 1.0; av=abs(v); best,bd=0.0,float("inf")
    for c in FP4_LEVELS:
        d=abs(av-c)
        if d<=bd: best,bd=c,d
    return s*best
def fp4_nib(v): return FP4_ENC[abs(round_fp4(v))]|(0x8 if math.copysign(1.0,v)<0 else 0)

def _eligible(g,min_n):
    const={n.id:n.attrs["weight"] for n in g.nodes if n.op==OpKind.CONSTANT}
    st={w:[0,0] for w in g.weights}
    for n in g.nodes:
        for pos,inp in enumerate(n.inputs):
            w=const.get(inp)
            if w is None: continue
            st[w][0]+=1
            if n.op in MATMUL_LIKE and pos==1: st[w][1]+=1
    return {name for name,(tot,rhs) in st.items() if tot>0 and tot==rhs and len(g.weights[name].shape)==2 and g.weights[name].numel()>=min_n and g.weights[name].dtype=="f32" and g.weights[name].quant=="none"}

class FP4Quant:
    BLOCK=32
    def run(self,g):
        for name in sorted(_eligible(g,self.BLOCK)): self._q(g.weights[name])
        return g
    def _q(self,b):
        flat=list(b.data); flat.extend([0.0]*((-len(flat))%self.BLOCK))
        nb=len(flat)//self.BLOCK; packed=bytearray(nb*16); scales=bytearray(nb); res=[]
        for blk in range(nb):
            s=blk*self.BLOCK; chunk=flat[s:s+self.BLOCK]; ac=[abs(x) for x in chunk]
            mean=sum(ac)/self.BLOCK; var=sum((x-mean)**2 for x in ac)/self.BLOCK; tau=mean+3.0*math.sqrt(var)
            non=[ac[i] for i in range(self.BLOCK) if ac[i]<=tau]; mx=max(non) if non else 0.0
            if mx<=0: mx=1e-8
            exp=max(-126,min(127,int(math.ceil(math.log2(mx/6.0))))); scale=2.0**exp; scales[blk]=(exp+127)&0xFF
            for i in range(self.BLOCK):
                v=chunk[i]
                if ac[i]>tau:
                    q=6.0 if v>=0 else -6.0; nib=fp4_nib(q)
                    if s+i<len(b.data): res.append((s+i,v-scale*q))
                else: nib=fp4_nib(v/scale)
                idx=blk*16+(i>>1); packed[idx]|=(nib&0x0F) if (i&1)==0 else ((nib&0x0F)<<4)
        cols=b.shape[1] if len(b.shape)==2 else 1
        res.sort(key=lambda item: (item[0]%cols, item[0]//cols))
        b.quant="fp4"; b.packed=bytes(packed); b.scales=bytes(scales); b.residuals=res

class TernaryQuant:
    MIN=4
    def run(self,g):
        for name in sorted(_eligible(g,self.MIN)): self._q(g.weights[name])
        return g
    def _q(self,b):
        data=b.data; n=len(data); alpha=max(1e-8,(sum(abs(x) for x in data)/n) if n else 1.0)
        packed=bytearray((n+3)//4)
        for i,v in enumerate(data):
            q=max(-1,min(1,_rha(v/alpha))); code=1 if q==1 else (2 if q==-1 else 0)
            packed[i>>2]|=code<<((i&3)*2)
        b.quant="ternary"; b.packed=bytes(packed); b.scale_f=float(alpha)

class Pipeline:
    def __init__(self,fusion=True,quant=True,ternary=False,brain=False):
        self.fusion=fusion; self.quant=quant; self.ternary=ternary; self.brain=brain
    def run(self,g):
        g=DCE().run(g)
        if self.brain: g=BrainRouterPass().run(g)
        if self.fusion: g=Fusion().run(g)
        if self.ternary: g=TernaryQuant().run(g)
        elif self.quant: g=FP4Quant().run(g)
        return DCE().run(g)
'''

FILES["axon/blob.py"] = r'''import struct
from typing import Dict
from .ir import WeightBlob
HEADER=struct.Struct("<4sIIIQ"); ENTRY=struct.Struct("<6Q4q")
MAGIC=b"AXN3"; VERSION=3; DTYPE_F32=0; DTYPE_FP4=2; DTYPE_TERNARY=4
def payload_bytes(b: WeightBlob) -> int:
    if b.quant=="fp4" and b.packed: return 8+len(b.scales or b"")+len(b.packed)+8*len(b.residuals)
    if b.quant=="ternary" and b.packed: return 8+len(b.packed)
    return len(b.data)*4
class BlobPacker:
    @staticmethod
    def _fnv(s):
        h=14695981039346656037
        for b in s.encode(): h=((h^b)*1099511628211)&0xFFFFFFFFFFFFFFFF
        return h
    @staticmethod
    def pack(weights:Dict[str,WeightBlob])->bytes:
        names=bytearray(); data=bytearray(); entries=[]
        for name in sorted(weights):
            b=weights[name]; enc=name.encode()+b"\0"; noff=len(names); names.extend(enc)
            while len(data)%16: data.append(0)
            doff=len(data)
            if b.quant=="fp4" and b.packed:
                sc=b.scales or b""; payload=bytearray(struct.pack("<II",len(sc),len(b.residuals))); payload.extend(sc); payload.extend(b.packed)
                for idx,diff in b.residuals: payload.extend(struct.pack("<If",int(idx),float(diff)))
                dtype=DTYPE_FP4
            elif b.quant=="ternary" and b.packed:
                payload=bytearray(struct.pack("<fI",float(b.scale_f),len(b.data))); payload.extend(b.packed); dtype=DTYPE_TERNARY
            else:
                payload=bytearray(struct.pack(f"<{len(b.data)}f",*b.data)); dtype=DTYPE_F32
            assert len(payload)==payload_bytes(b)
            data.extend(payload); shp=(list(b.shape)+[0,0,0,0])[:4]
            entries.append((BlobPacker._fnv(name),noff,doff,len(payload),dtype,len(b.shape),shp))
        count=len(entries); base=HEADER.size+count*ENTRY.size+len(names); pad=(-base)%16; dbase=base+pad; total=dbase+len(data)
        out=bytearray(HEADER.pack(MAGIC,VERSION,count,0,total))
        for h,noff,doff,nb,dt,nd,shp in entries: out.extend(ENTRY.pack(h,noff,dbase+doff,nb,dt,nd,*shp))
        out.extend(names); out.extend(b"\0"*pad); out.extend(data); assert len(out)==total; return bytes(out)
'''

FILES["axon/reference.py"] = r'''import math
from .ir import OpKind, MATMUL_LIKE
FP4_LUT=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]
class T:
    __slots__=("shape","data")
    def __init__(self,shape,data): self.shape=list(shape); self.data=list(data)
    def numel(self):
        n=1
        for d in self.shape: n*=d
        return n
    def strides(self):
        s=[1]*len(self.shape)
        for i in range(len(self.shape)-2,-1,-1): s[i]=s[i+1]*self.shape[i+1]
        return s
def dequant(blob):
    if blob.quant=="ternary":
        return [blob.scale_f if ((blob.packed[i>>2]>>((i&3)*2))&3)==1 else (-blob.scale_f if ((blob.packed[i>>2]>>((i&3)*2))&3)==2 else 0.0) for i in range(len(blob.data))]
    if blob.quant=="fp4":
        out=[0.0]*len(blob.data)
        for b in range(len(blob.scales)):
            sc=2.0**(blob.scales[b]-127)
            for i in range(32):
                byte=blob.packed[b*16+i//2]; nib=byte&0x0F if i%2==0 else (byte>>4)&0x0F; idx=b*32+i
                if idx<len(out): out[idx]=FP4_LUT[nib]*sc
        for idx,diff in blob.residuals: out[idx]+=diff
        return out
    return list(blob.data)
def _bshape(a,b):
    n=max(len(a),len(b)); pa=[1]*(n-len(a))+list(a); pb=[1]*(n-len(b))+list(b); out=[]
    for x,y in zip(pa,pb):
        if x==y: out.append(x)
        elif x==1: out.append(y)
        elif y==1: out.append(x)
        else: raise ValueError(f"broadcast {a} {b}")
    return out
def _bstrides(t,shape):
    st=[0]*len(shape); off=len(shape)-len(t.shape); ts=t.strides()
    for i,d in enumerate(t.shape): st[off+i]=ts[i] if d==shape[off+i] else 0
    return st
def _ew(a,b,mode):
    sh=_bshape(a.shape,b.shape); out=T(sh,[0.0]*math.prod(sh)); os_=out.strides(); sa=_bstrides(a,sh); sb=_bstrides(b,sh)
    for lin in range(out.numel()):
        rem=lin; oa=ob=0
        for d in range(len(sh)): idx=rem//os_[d]; rem-=idx*os_[d]; oa+=idx*sa[d]; ob+=idx*sb[d]
        x,y=a.data[oa],b.data[ob]; out.data[lin]=x+y if mode==0 else (x-y if mode==1 else x*y)
    return out
def _mm(a,b):
    M,K=a.shape; K2,N=b.shape; assert K==K2
    return T([M,N],[sum(a.data[i*K+k]*b.data[k*N+j] for k in range(K)) for i in range(M) for j in range(N)])
def _neuro_mm(x,w,s):
    M,K=x.shape; K2,N=w.shape; K3,R=s.shape; assert K==K2 and K==K3
    keep=max(1,R//2); out=T([M,N],[0.0]*(M*N))
    for i in range(M):
        scores=[abs(sum(x.data[i*K+k]*s.data[k*R+r] for k in range(K))) for r in range(R)]
        for r in sorted(range(R),key=lambda r:(-scores[r],r))[:keep]:
            c0,c1=(r*N)//R,((r+1)*N)//R
            for j in range(c0,c1): out.data[i*N+j]=sum(x.data[i*K+k]*w.data[k*N+j] for k in range(K))
    return out
def sigmoid(v):
    if v>=0.0: return 1.0/(1.0+math.exp(-v))
    e=math.exp(v); return e/(1.0+e)
def silu(v): return v*sigmoid(v)
def gelu(v): return 0.5*v*(1.0+math.tanh(0.7978845608028654*(v+0.044715*v*v*v)))
def _act(t,k):
    if k=="relu": return T(t.shape,[v if v>0 else 0.0 for v in t.data])
    if k=="silu": return T(t.shape,[silu(v) for v in t.data])
    if k=="sigmoid": return T(t.shape,[sigmoid(v) for v in t.data])
    return T(t.shape,[gelu(v) for v in t.data])
def _softmax(t):
    cols=t.shape[-1]; rows=t.numel()//cols; out=T(t.shape,[0.0]*t.numel())
    for r in range(rows):
        row=t.data[r*cols:(r+1)*cols]; mx=max(row); e=[math.exp(v-mx) for v in row]; s=sum(e)
        out.data[r*cols:(r+1)*cols]=[v/s for v in e]
    return out
def _ln(x,g,b):
    cols=x.shape[-1]; rows=x.numel()//cols; out=T(x.shape,[0.0]*x.numel())
    for r in range(rows):
        row=x.data[r*cols:(r+1)*cols]; mu=sum(row)/cols; var=sum((v-mu)**2 for v in row)/cols; inv=1.0/math.sqrt(var+1e-5)
        out.data[r*cols:(r+1)*cols]=[((row[j]-mu)*inv)*g.data[j]+b.data[j] for j in range(cols)]
    return out
def _transpose(t):
    M,N=t.shape; return T([N,M],[t.data[i*N+j] for j in range(N) for i in range(M)])
def _eq(x,w,b):
    D=x.shape[-1]; M=x.numel()//D; s=list(x.data)
    for _ in range(16):
        sn=[0.0]*len(s); md=0.0
        for i in range(M):
            for j in range(D):
                acc=x.data[i*D+j]+b.data[j]+sum(s[i*D+k]*w.data[k*D+j] for k in range(D))
                v=0.7*s[i*D+j]+0.3*math.tanh(acc); md=max(md,abs(v-s[i*D+j])); sn[i*D+j]=v
        s=sn
        if md<1e-4: break
    return T(x.shape,s)
def evaluate(graph, input_tensors):
    vals={}
    for idx,iid in enumerate(graph.inputs): vals[iid]=input_tensors[idx]
    for n in graph.topo():
        if n.op==OpKind.INPUT: continue
        if n.op==OpKind.CONSTANT: b=graph.weights[n.attrs["weight"]]; vals[n.id]=T(b.shape,dequant(b)); continue
        a=[vals[i] for i in n.inputs]
        if n.op==OpKind.NEURO_MATMUL: r=_neuro_mm(a[0],a[1],a[2])
        elif n.op in MATMUL_LIKE:
            r=_mm(a[0],a[1])
            if n.op in (OpKind.FUSED_MATMUL_BIAS,OpKind.FUSED_MATMUL_BIAS_RELU,OpKind.FUSED_MATMUL_BIAS_GELU,OpKind.FUSED_MATMUL_BIAS_SILU): r=_ew(r,a[2],0)
            if n.op in (OpKind.FUSED_MATMUL_RELU,OpKind.FUSED_MATMUL_BIAS_RELU): r=_act(r,"relu")
            elif n.op in (OpKind.FUSED_MATMUL_GELU,OpKind.FUSED_MATMUL_BIAS_GELU): r=_act(r,"gelu")
            elif n.op in (OpKind.FUSED_MATMUL_SILU,OpKind.FUSED_MATMUL_BIAS_SILU): r=_act(r,"silu")
        elif n.op==OpKind.ADD: r=_ew(a[0],a[1],0)
        elif n.op==OpKind.SUB: r=_ew(a[0],a[1],1)
        elif n.op==OpKind.MUL: r=_ew(a[0],a[1],2)
        elif n.op==OpKind.RELU: r=_act(a[0],"relu")
        elif n.op==OpKind.GELU: r=_act(a[0],"gelu")
        elif n.op==OpKind.SILU: r=_act(a[0],"silu")
        elif n.op==OpKind.SIGMOID: r=_act(a[0],"sigmoid")
        elif n.op==OpKind.SOFTMAX: r=_softmax(a[0])
        elif n.op==OpKind.LAYERNORM: r=_ln(a[0],a[1],a[2])
        elif n.op==OpKind.TRANSPOSE: r=_transpose(a[0])
        elif n.op==OpKind.EQUILIBRIUM: r=_eq(a[0],a[1],a[2])
        else: raise ValueError(f"reference: unhandled {n.op}")
        vals[n.id]=r
    return vals[graph.outputs[0]]
'''

FILES["axon/backend.py"] = r'''import os, platform, shutil, subprocess, tempfile
import llvmlite.ir as ir
import llvmlite.binding as llvm
from .errors import CodegenError
from .ir import OpKind, MATMUL_LIKE
from .blob import BlobPacker
from .runtime_paths import runtime_paths

_INIT=False
def _init_llvm():
    global _INIT
    if _INIT: return
    try: llvm.initialize_all_targets(); llvm.initialize_all_asmprinters()
    except AttributeError: llvm.initialize(); llvm.initialize_native_target(); llvm.initialize_native_asmprinter()
    _INIT=True
def _numel(sh):
    n=1
    for d in sh: n*=d
    return n
def estimate_flops(g):
    t=0
    for n in g.nodes:
        if n.op==OpKind.NEURO_MATMUL:
            a,b,s=g.get(n.inputs[0]),g.get(n.inputs[1]),g.get(n.inputs[2])
            if a and b and s and len(a.shape)==2 and len(b.shape)==2: t+=2*a.shape[0]*a.shape[1]*s.shape[1]+a.shape[0]*a.shape[1]*b.shape[1]
        elif n.op in MATMUL_LIKE:
            a,b=g.get(n.inputs[0]),g.get(n.inputs[1])
            if a and b and len(a.shape)==2 and len(b.shape)==2:
                m,k=a.shape; _,nn=b.shape; t+=2*m*k*nn
                if n.op!=OpKind.MATMUL: t+=m*nn*(5 if ("GELU" in n.op.name or "SILU" in n.op.name) else 2)
        elif n.op==OpKind.EQUILIBRIUM: t+=_numel(n.shape)*16*(2*(n.shape[-1] if n.shape else 1)+5)
        elif n.op in (OpKind.ADD,OpKind.SUB,OpKind.MUL,OpKind.TRANSPOSE,OpKind.RELU): t+=_numel(n.shape)
        elif n.op in (OpKind.GELU,OpKind.SILU,OpKind.SIGMOID): t+=_numel(n.shape)*8
        elif n.op in (OpKind.SOFTMAX,OpKind.LAYERNORM): t+=_numel(n.shape)*6
    return t
def _tool(name,target=""):
    if target and target!="native":
        c=f"{target}-{name}"
        if shutil.which(c): return c
    if name=="cc": return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang") or "cc"
    if name=="ar": return os.environ.get("AR") or shutil.which("ar") or "ar"
    return name
def _san_flags(): return ["-fsanitize=address,undefined","-fno-omit-frame-pointer"] if os.environ.get("AXON_SANITIZE","0")=="1" else []
def base_cflags(native=False):
    f=["-O1" if _san_flags() else "-O3","-fPIC","-std=c11","-Wall","-Wextra","-Wno-unknown-pragmas","-fno-math-errno","-pthread"]
    if _san_flags(): f+=["-g",*_san_flags()]
    if native and platform.machine().lower() in ("x86_64","amd64","aarch64","arm64"): f.append("-march=native")
    return f

class Linker:
    @staticmethod
    def compile_c(src,obj,target="",native=False,extra=None):
        r=subprocess.run([_tool("cc",target),*base_cflags(native),*(extra or []),"-c",src,"-o",obj],capture_output=True,text=True)
        if r.returncode!=0: raise CodegenError("C compile failed:\n"+r.stderr.strip())
        return obj
    @staticmethod
    def archive(objects,out_a,target=""):
        r=subprocess.run([_tool("ar",target),"rcs",out_a,*[o for o in objects if not o.endswith(".a")]],capture_output=True,text=True)
        if r.returncode!=0: raise CodegenError("ar failed:\n"+r.stderr.strip())
        return out_a
    @staticmethod
    def link(objects,out_bin,static_rt="",target="",native=False,omp=None):
        from .tools import Inspector
        if omp is None: omp=Inspector.shared().omp_ldflags if not target else ["-fopenmp"]
        cmd=[_tool("cc",target),"-O1" if _san_flags() else "-O3","-pthread",*_san_flags(),*omp,*objects]
        if static_rt: cmd.append(static_rt)
        cmd.extend(["-lm","-o",out_bin])
        r=subprocess.run(cmd,capture_output=True,text=True)
        if r.returncode!=0: raise CodegenError("link failed:\n"+r.stderr.strip())
        os.chmod(out_bin,0o755); return out_bin

class LLVMBackend:
    def __init__(self,module_name="axon",target_triple=""):
        _init_llvm(); self.mod=ir.Module(module_name); self.triple=target_triple or llvm.get_default_triple(); self.mod.triple=self.triple
        self.void=ir.VoidType(); self.i8=ir.IntType(8); self.i32=ir.IntType(32); self.i64=ir.IntType(64)
        self.i8p=ir.PointerType(self.i8); self.i64p=ir.PointerType(self.i64); self.i8pp=ir.PointerType(self.i8p)
        self._strs={}; self._declare()
    def _f(self,name,ret,args): return ir.Function(self.mod,ir.FunctionType(ret,args),name=name)
    def _declare(self):
        p,i32,i64,v=self.i8p,self.i32,self.i64,self.void
        self.f_alloc=self._f("axon_tensor_alloc",p,[i64,self.i64p,i32]); self.f_free=self._f("axon_tensor_free",v,[p])
        self.f_weight=self._f("axon_weight_tensor",p,[p]); self.f_setblob=self._f("axon_set_weights_blob",v,[p,i64])
        self.f_wget=self._f("axon_weights_get",v,[self.i8pp,self.i64p]); self.f_lock=self._f("axon_rt_lock",v,[]); self.f_unlock=self._f("axon_rt_unlock",v,[])
        b3=[p,p,p]; b4=[p,p,p,p]
        self.kernels={
            OpKind.MATMUL:(self._f("axon_matmul",i32,b3),2), OpKind.NEURO_MATMUL:(self._f("axon_neuro_matmul",i32,b4),3),
            OpKind.ADD:(self._f("axon_add",i32,b3),2), OpKind.SUB:(self._f("axon_sub",i32,b3),2), OpKind.MUL:(self._f("axon_mul",i32,b3),2),
            OpKind.RELU:(self._f("axon_relu",i32,[p,p]),1), OpKind.GELU:(self._f("axon_gelu",i32,[p,p]),1),
            OpKind.SILU:(self._f("axon_silu",i32,[p,p]),1), OpKind.SIGMOID:(self._f("axon_sigmoid",i32,[p,p]),1),
            OpKind.SOFTMAX:(self._f("axon_softmax",i32,[p,p]),1), OpKind.TRANSPOSE:(self._f("axon_transpose",i32,[p,p]),1),
            OpKind.LAYERNORM:(self._f("axon_layer_norm",i32,b4),3), OpKind.EQUILIBRIUM:(self._f("axon_equilibrium",i32,b4),3),
            OpKind.FUSED_MATMUL_RELU:(self._f("axon_mm_relu",i32,b3),2), OpKind.FUSED_MATMUL_GELU:(self._f("axon_mm_gelu",i32,b3),2),
            OpKind.FUSED_MATMUL_SILU:(self._f("axon_mm_silu",i32,b3),2), OpKind.FUSED_MATMUL_BIAS:(self._f("axon_mm_bias",i32,b4),3),
            OpKind.FUSED_MATMUL_BIAS_RELU:(self._f("axon_mm_bias_relu",i32,b4),3), OpKind.FUSED_MATMUL_BIAS_GELU:(self._f("axon_mm_bias_gelu",i32,b4),3),
            OpKind.FUSED_MATMUL_BIAS_SILU:(self._f("axon_mm_bias_silu",i32,b4),3),
        }
    def _cstr(self,b,text):
        gv=self._strs.get(text)
        if gv is None:
            raw=bytearray(text.encode()+b"\0"); ty=ir.ArrayType(self.i8,len(raw))
            gv=ir.GlobalVariable(self.mod,ty,name=f".str.{len(self._strs)}"); gv.linkage="internal"; gv.global_constant=True
            gv.initializer=ir.Constant(ty,raw); self._strs[text]=gv
        z=ir.Constant(self.i32,0); return b.gep(gv,[z,z],inbounds=True)
    def _shape_ptr(self,b,shape):
        arr=b.alloca(ir.ArrayType(self.i64,max(len(shape),1))); z=ir.Constant(self.i32,0)
        for i,d in enumerate(shape): b.store(ir.Constant(self.i64,int(d)),b.gep(arr,[z,ir.Constant(self.i32,i)],inbounds=True))
        return b.gep(arr,[z,z],inbounds=True)
    def _alloc(self,b,shape): return b.call(self.f_alloc,[ir.Constant(self.i64,len(shape)),self._shape_ptr(b,shape),ir.Constant(self.i32,0)])
    def build(self,g,baked=True):
        if not g.outputs or len(g.outputs)!=1: raise CodegenError("graph needs exactly one output")
        if not g.inputs: raise CodegenError("graph has no inputs")
        g.reindex(); order=g.topo(); names=sorted(g.weights); slots={n:i for i,n in enumerate(names)}
        nw=max(len(names),1); at=ir.ArrayType(self.i8p,nw)
        self.g_w=ir.GlobalVariable(self.mod,at,"axon_w_cache"); self.g_w.linkage="internal"; self.g_w.initializer=ir.Constant(at,[ir.Constant(self.i8p,None)]*nw)
        self.g_init=ir.GlobalVariable(self.mod,self.i32,"axon_w_ready"); self.g_init.linkage="internal"; self.g_init.initializer=ir.Constant(self.i32,0)
        self._emit_init(names,baked); self._emit_shutdown(len(names)); self._emit_forward(g,order,slots); self._emit_meta(g); return self.mod
    def _emit_init(self,names,baked):
        z=ir.Constant(self.i32,0); one=ir.Constant(self.i32,1)
        fn=ir.Function(self.mod,ir.FunctionType(self.i32,[]),"axon_init")
        entry=fn.append_basic_block("entry"); work=fn.append_basic_block("work"); done=fn.append_basic_block("done")
        b=ir.IRBuilder(entry); b.call(self.f_lock,[]); ready=b.load(self.g_init); b.cbranch(b.icmp_unsigned("==",ready,z),work,done)
        b.position_at_end(work)
        if baked:
            pslot=b.alloca(self.i8p); nslot=b.alloca(self.i64); b.call(self.f_wget,[pslot,nslot])
            p=b.load(pslot); n=b.load(nslot)
            has_b=fn.append_basic_block("setb"); cont=fn.append_basic_block("cont")
            b.cbranch(b.icmp_unsigned("!=",p,ir.Constant(self.i8p,None)),has_b,cont)
            b.position_at_end(has_b); b.call(self.f_setblob,[p,n]); b.branch(cont)
            b.position_at_end(cont)
        st=z
        for k,name in enumerate(names):
            ptr=b.call(self.f_weight,[self._cstr(b,name)]); b.store(ptr,b.gep(self.g_w,[z,ir.Constant(self.i32,k)],inbounds=True))
            st=b.or_(st,b.zext(b.icmp_unsigned("==",ptr,ir.Constant(self.i8p,None)),self.i32))
        b.store(b.select(b.icmp_unsigned("==",st,z),one,z),self.g_init); work_end=b.block; b.branch(done)
        b.position_at_end(done); phi=b.phi(self.i32); phi.add_incoming(z,entry); phi.add_incoming(st,work_end); b.call(self.f_unlock,[]); b.ret(phi)
    def _emit_shutdown(self,count):
        z=ir.Constant(self.i32,0); fn=ir.Function(self.mod,ir.FunctionType(self.void,[]),"axon_shutdown"); b=ir.IRBuilder(fn.append_basic_block("e")); b.call(self.f_lock,[])
        for k in range(count):
            slot=b.gep(self.g_w,[z,ir.Constant(self.i32,k)],inbounds=True); b.call(self.f_free,[b.load(slot)]); b.store(ir.Constant(self.i8p,None),slot)
        b.store(z,self.g_init); b.call(self.f_unlock,[]); b.ret_void()
    def _emit_forward(self,g,order,slots):
        z=ir.Constant(self.i32,0); null=ir.Constant(self.i8p,None)
        fn=ir.Function(self.mod,ir.FunctionType(self.i8p,[self.i8pp]),"axon_forward")
        entry=fn.append_basic_block("entry"); body=fn.append_basic_block("body"); fail=fn.append_basic_block("fail"); cleanup=fn.append_basic_block("cleanup"); ok=fn.append_basic_block("ok")
        compute=[n for n in order if n.op not in (OpKind.INPUT,OpKind.CONSTANT)]; out_id=g.outputs[0]; temps={n.id for n in compute}
        b=ir.IRBuilder(entry); live_slots={nid:b.alloca(self.i8p) for nid in sorted(temps)}
        for sp in live_slots.values(): b.store(null,sp)
        rc=b.call(self.f_init,[]); b.cbranch(b.icmp_unsigned("==",rc,z),body,fail)
        b.position_at_end(fail); b.ret(null); b.position_at_end(body)
        vals={}; [vals.__setitem__(iid,b.load(b.gep(fn.args[0],[ir.Constant(self.i32,idx)],inbounds=True))) for idx,iid in enumerate(g.inputs)]
        for name,slot in slots.items(): vals[("w",name)]=b.load(b.gep(self.g_w,[z,ir.Constant(self.i32,slot)],inbounds=True))
        def operand(nid):
            n=g.get(nid); return vals[("w",n.attrs["weight"])] if n.op==OpKind.CONSTANT else vals[nid]
        last={}; [last.__setitem__(inp,i) for i,n in enumerate(compute) for inp in n.inputs]; status=z
        for i,n in enumerate(compute):
            kern,_=self.kernels[n.op]; out=self._alloc(b,n.shape); b.store(out,live_slots[n.id])
            status=b.or_(status,b.call(kern,[operand(x) for x in n.inputs]+[out])); vals[n.id]=out
            for pid,lu in last.items():
                if lu==i and pid!=out_id and pid in temps: b.call(self.f_free,[vals[pid]]); b.store(null,live_slots[pid])
        result=operand(out_id); b.cbranch(b.icmp_unsigned("!=",status,z),cleanup,ok)
        b.position_at_end(cleanup)
        for nid in sorted(temps): b.call(self.f_free,[b.load(live_slots[nid])])
        b.ret(null); b.position_at_end(ok); b.ret(result)
    def _emit_meta(self,g):
        first=g.get(g.inputs[0]); shape=tuple(first.shape); n=_numel(shape)
        f=ir.Function(self.mod,ir.FunctionType(self.i64,[]),"axon_get_total_flops"); ir.IRBuilder(f.append_basic_block("e")).ret(ir.Constant(self.i64,estimate_flops(g)))
        f=ir.Function(self.mod,ir.FunctionType(self.i64,[]),"axon_get_input_ndim"); ir.IRBuilder(f.append_basic_block("e")).ret(ir.Constant(self.i64,len(shape)))
        f=ir.Function(self.mod,ir.FunctionType(self.i64,[]),"axon_get_input_numel"); ir.IRBuilder(f.append_basic_block("e")).ret(ir.Constant(self.i64,n))
        f=ir.Function(self.mod,ir.FunctionType(self.i64,[]),"axon_output_is_temp"); ir.IRBuilder(f.append_basic_block("e")).ret(ir.Constant(self.i64,1 if g.get(g.outputs[0]).op not in (OpKind.INPUT,OpKind.CONSTANT) else 0))
        f=ir.Function(self.mod,ir.FunctionType(self.void,[self.i64p]),"axon_get_input_shape"); b=ir.IRBuilder(f.append_basic_block("e"))
        for i,d in enumerate(shape): b.store(ir.Constant(self.i64,int(d)),b.gep(f.args[0],[ir.Constant(self.i32,i)],inbounds=True))
        b.ret_void()
    def compile(self,g,out_obj,baked=True):
        mod=self.build(g,baked); parsed=llvm.parse_assembly(str(mod)); parsed.verify()
        try: target=llvm.Target.from_triple(self.triple)
        except RuntimeError as e: raise CodegenError(f"bad triple {self.triple!r}: {e}")
        tm=target.create_target_machine(opt=3)
        with open(out_obj,"wb") as f: f.write(tm.emit_object(parsed))
        return out_obj
    def parse_for_jit(self,g):
        mod=self.build(g,baked=False); parsed=llvm.parse_assembly(str(mod)); parsed.verify(); return parsed

EMBED_HDR="""/* Auto-generated by Axon v7.0. */
#include <stdint.h>
#if defined(_MSC_VER)
__declspec(align(16)) static const unsigned char blob[%d] = {
%s
};
#else
static const unsigned char blob[%d] __attribute__((aligned(16))) = {
%s
};
#endif
void axon_weights_get(const void **p, int64_t *n) { *p = blob; *n = (int64_t)%d; }
"""
EXT_STUB="#include <stdint.h>\n#include <stddef.h>\nvoid axon_weights_get(const void **p, int64_t *n) { *p = NULL; *n = 0; }\n"
def _c_hex_array(blob):
    b=bytes(blob)
    if not b: raise CodegenError("internal: empty weight blob")
    return ",\n".join("    "+", ".join(f"0x{x:02x}" for x in b[i:i+16]) for i in range(0,len(b),16))

class Baker:
    def __init__(self,target="",native=False): self.target=target; self.native=native
    def bake(self,objects,weights,output,emit_lib=False,external_weights=False):
        rt=runtime_paths(); blob=BlobPacker.pack(weights)
        with tempfile.TemporaryDirectory(prefix="axon_") as tmp:
            csrc=os.path.join(tmp,"w.c")
            if external_weights and not emit_lib:
                with open(output+".axn3","wb") as wf: wf.write(blob)
                with open(csrc,"w",encoding="ascii") as f: f.write(EXT_STUB)
            else:
                lit=_c_hex_array(blob); sz=len(blob)
                with open(csrc,"w",encoding="ascii") as f: f.write(EMBED_HDR%(sz,lit,sz,lit,sz))
            wobj=os.path.join(tmp,"w.o"); Linker.compile_c(csrc,wobj,self.target,self.native)
            if emit_lib:
                lib=output if output.endswith(".a") else output+".a"; hdr=lib[:-2]+".h"
                Linker.archive([*objects,wobj,str(rt.core_obj)],lib,self.target)
                with open(hdr,"w") as f:
                    f.write('#ifndef AXON_EMBED_H\n#define AXON_EMBED_H\n#include "axon_runtime.h"\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
                            'int axon_init(void);\nvoid axon_shutdown(void);\nAxonTensor *axon_forward(AxonTensor **inputs);\n'
                            'int64_t axon_get_input_ndim(void);\nvoid axon_get_input_shape(int64_t *out);\n'
                            'int64_t axon_get_input_numel(void);\nint64_t axon_get_total_flops(void);\nint64_t axon_output_is_temp(void);\n'
                            '#ifdef __cplusplus\n}\n#endif\n#endif\n')
                return lib
            Linker.link([*objects,wobj,str(rt.main_obj)],output,str(rt.static_lib),self.target,self.native); return output

CORE=r"""`timescale 1ns/1ps
module axon_mac_core #(
    parameter integer DATA_W=16, parameter integer FRAC_W=%(FRAC_W)d,
    parameter integer K_DIM=%(K)d, parameter integer USE_RELU=%(RELU)d
)(
    input wire clk, rst_n, start, in_valid,
    input wire signed [DATA_W-1:0] in_val, weight_val,
    output reg signed [DATA_W-1:0] out_val, output reg done, busy
);
    localparam integer ACC_W=2*DATA_W+8;
    localparam integer SH_AMT=(FRAC_W>0)?(FRAC_W-1):0;
    localparam signed [ACC_W-1:0] ONE=1;
    localparam signed [ACC_W-1:0] MAX_OUT=(ONE<<<(DATA_W-1))-1;
    localparam signed [ACC_W-1:0] MIN_OUT=-(ONE<<<(DATA_W-1));
    localparam signed [ACC_W-1:0] ROUND_BIAS=(FRAC_W>0)?(ONE<<<SH_AMT):{ACC_W{1'b0}};
    localparam [1:0] S_IDLE=2'd0, S_MAC=2'd1, S_ACT=2'd2, S_DONE=2'd3;
    reg [1:0] state; reg [31:0] cnt; reg signed [ACC_W-1:0] acc;
    wire signed [ACC_W-1:0] scaled_acc=(FRAC_W>0)?((acc+ROUND_BIAS)>>>FRAC_W):acc;
    initial if (K_DIM<1 || DATA_W<2 || FRAC_W<0) begin $display("invalid params"); $finish(1); end
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin state<=S_IDLE; cnt<=0; acc<=0; out_val<=0; done<=0; busy<=0; end
        else case(state)
            S_IDLE: begin done<=0; if(start) begin acc<=0; cnt<=0; busy<=1; state<=S_MAC; end end
            S_MAC: if(in_valid) begin acc<=acc+(in_val*weight_val); if(cnt==K_DIM-1) state<=S_ACT; else cnt<=cnt+1; end
            S_ACT: begin
                if((USE_RELU!=0)&&(scaled_acc<=0)) out_val<=0;
                else if(scaled_acc>MAX_OUT) out_val<=MAX_OUT[DATA_W-1:0];
                else if(scaled_acc<MIN_OUT) out_val<=MIN_OUT[DATA_W-1:0];
                else out_val<=scaled_acc[DATA_W-1:0];
                done<=1; busy<=0; state<=S_DONE; end
            S_DONE: if(!start) begin done<=0; state<=S_IDLE; end
            default: state<=S_IDLE;
        endcase
    end
endmodule
"""
TB=r"""`timescale 1ns/1ps
module tb_axon_mac_core;
    localparam W=16, F=%(FRAC_W)d, K=%(K)d, RELU=%(RELU)d;
    localparam integer ACC_W=2*W+8; localparam integer SH_AMT=(F>0)?(F-1):0;
    localparam signed [ACC_W-1:0] ONE=1, MAX_OUT=(ONE<<<(W-1))-1, MIN_OUT=-(ONE<<<(W-1));
    localparam signed [ACC_W-1:0] ROUND_BIAS=(F>0)?(ONE<<<SH_AMT):{ACC_W{1'b0}};
    reg clk=0, rst_n=0, start=0, in_valid=0; reg signed [W-1:0] a, w;
    wire signed [W-1:0] y; wire done, busy; integer i;
    reg signed [W-1:0] av[0:K-1], wv[0:K-1], rom_w[0:K-1];
    reg signed [ACC_W-1:0] gacc, gscaled; reg signed [W-1:0] gexp;
    axon_mac_core #(.DATA_W(W),.FRAC_W(F),.K_DIM(K),.USE_RELU(RELU)) dut(.clk(clk),.rst_n(rst_n),.start(start),.in_valid(in_valid),.in_val(a),.weight_val(w),.out_val(y),.done(done),.busy(busy));
    always #5 clk=~clk;
    task run_tx(input signed [W-1:0] exp_y);
        begin
            @(negedge clk); start=1;
            for(i=0;i<K;i=i+1) begin @(negedge clk); in_valid=1; a=av[i]; w=wv[i]; end
            @(negedge clk); in_valid=0; start=0; wait(done); @(negedge clk);
            if(y!==exp_y) begin $display("FAIL got=%%0d exp=%%0d", y, exp_y); $finish(1); end
            @(posedge clk); #1;
            if(done!==0 || busy!==0) begin $display("FAIL S_DONE->S_IDLE"); $finish(1); end
        end
    endtask
    initial begin
        $readmemh("%(HEX)s", rom_w); #12 rst_n=1;
        gacc=0; for(i=0;i<K;i=i+1) begin av[i]=16'sd256; wv[i]=rom_w[i]; gacc=gacc+(av[i]*wv[i]); end
        gscaled=(F>0)?((gacc+ROUND_BIAS)>>>F):gacc;
        gexp=((RELU!=0)&&(gscaled<=0))?0:((gscaled>MAX_OUT)?MAX_OUT[W-1:0]:((gscaled<MIN_OUT)?MIN_OUT[W-1:0]:gscaled[W-1:0]));
        run_tx(gexp);
        gacc=0; for(i=0;i<K;i=i+1) begin av[i]=-16'sd512; wv[i]=16'sd256; gacc=gacc+(av[i]*wv[i]); end
        gscaled=(F>0)?((gacc+ROUND_BIAS)>>>F):gacc;
        gexp=((RELU!=0)&&(gscaled<=0))?0:((gscaled>MAX_OUT)?MAX_OUT[W-1:0]:((gscaled<MIN_OUT)?MIN_OUT[W-1:0]:gscaled[W-1:0]));
        run_tx(gexp);
        $display("PASS axon_mac_core multi-tx Q8.8 + $readmemh verified (K=%0d)", K); $finish;
    end
endmodule
"""
class Verilog:
    @staticmethod
    def compile(g,out_path,with_tb=True,frac_w=8):
        k,relu,wvals=4,1,[]
        for n in g.topo():
            if n.op in MATMUL_LIKE:
                a=g.get(n.inputs[0]); b=g.get(n.inputs[1])
                if a and len(a.shape)==2: k=int(a.shape[1])
                if b and b.op==OpKind.CONSTANT:
                    wb=g.weights.get(b.attrs["weight"])
                    if wb and len(wb.shape)==2:
                        rows,cols=wb.shape; kk=min(k,rows)
                        wvals=[wb.data[i*cols] for i in range(kk)]+[0.0]*(k-kk)
                relu=1 if n.op in (OpKind.FUSED_MATMUL_RELU,OpKind.FUSED_MATMUL_BIAS_RELU) else 0
                break
        k=max(1,k)
        if not wvals: wvals=[0.25]*k
        with open(out_path,"w") as f: f.write(CORE%{"K":k,"RELU":relu,"FRAC_W":frac_w})
        stem=out_path.rsplit(".",1)[0]; hex_path=stem+"_weights.hex"; scale=1<<frac_w
        with open(hex_path,"w") as f:
            for v in wvals: f.write(f"{max(-32768,min(32767,int(round(v*scale))))&0xFFFF:04x}\n")
        if with_tb:
            abs_hex=os.path.abspath(hex_path).replace("\\","/")
            with open(stem+"_tb.v","w") as f: f.write(TB%{"K":k,"RELU":relu,"FRAC_W":frac_w,"HEX":abs_hex})
        return out_path
'''

FILES["axon/runtime_paths.py"] = r'''import os, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from .errors import CodegenError
PKG=Path(__file__).resolve().parent
@dataclass(frozen=True)
class RuntimePaths: dir:Path; static_lib:Path; shared_lib:Path; main_obj:Path; core_obj:Path; header:Path
def runtime_dir(): return Path(os.environ.get("AXON_RUNTIME_DIR",str(PKG/"runtime"))).resolve()
def tuning_header_path():
    rt=runtime_dir()/"axon_tuning.h"
    if rt.exists(): return rt
    c=Path(os.environ.get("AXON_CACHE_DIR",str(Path.home()/".cache"/"axon")))/"axon_tuning.h"
    return c if c.exists() else None
def runtime_paths():
    d=runtime_dir()
    return RuntimePaths(d,d/"libaxon_runtime.a",d/"libaxon_runtime.so",d/"axon_main.o",d/"axon_runtime.o",d/"axon_runtime.h")
def _stamp(rt): return rt.dir/".axon_tuning_stamp"
def _want_stamp():
    th=tuning_header_path()
    return f"{str(th) if th is not None else ''}|san={int(os.environ.get('AXON_SANITIZE','0')=='1')}"
def _stale(rt):
    outs=[rt.static_lib,rt.shared_lib,rt.main_obj,rt.core_obj]
    if not all(p.exists() for p in outs): return True
    sp=_stamp(rt); have=sp.read_text().strip() if sp.exists() else ""
    if _want_stamp()!=have: return True
    oldest=min(p.stat().st_mtime for p in outs); th=tuning_header_path()
    sources=list(rt.dir.glob("*.c"))+list(rt.dir.glob("*.h"))+[rt.dir/"Makefile"]+([th] if th is not None else [])
    return any(s.exists() and s.stat().st_mtime>oldest for s in sources)
def ensure_runtime(verbose=False,native=None,extra=None):
    from .tools import Inspector
    rt=runtime_paths()
    if not rt.dir.exists(): raise CodegenError(f"runtime dir missing: {rt.dir}")
    if extra is None and not _stale(rt): return rt
    insp=Inspector.shared()
    if native is None: native=os.environ.get("AXON_NATIVE","0")=="1"
    san=os.environ.get("AXON_SANITIZE","0")=="1"
    env=os.environ.copy(); env["NATIVE"]="1" if native else "0"; env["SANITIZE"]="1" if san else "0"
    env["OMP_CFLAGS"]=" ".join(insp.omp_cflags); env["OMP_LDFLAGS"]=" ".join(insp.omp_ldflags)
    th=tuning_header_path(); ex=[f"-I{th.parent}"] if th is not None else []
    if extra: ex.append(extra)
    if ex: env["EXTRA"]=" ".join(ex)
    if verbose: print(f"[axon] building runtime in {rt.dir} (san={int(san)})",file=sys.stderr)
    r=subprocess.run(["make","-C",str(rt.dir),"all"],env=env,capture_output=True,text=True)
    if r.returncode!=0: raise CodegenError("runtime build failed:\n"+(r.stderr or r.stdout).strip())
    if extra is None:
        try: _stamp(rt).write_text(_want_stamp()+"\n")
        except OSError: pass
    return rt
'''

FILES["axon/tools.py"] = r'''import json, math, os, platform, random, shutil, struct, subprocess, sys, tempfile
from array import array
from pathlib import Path
from .errors import AxonError
from .runtime_paths import runtime_dir

class Inspector:
    _shared=None
    @classmethod
    def shared(cls):
        if cls._shared is None: i=cls(); i.inspect(); cls._shared=i
        return cls._shared
    def __init__(self):
        self.os_name=platform.system().lower(); self.arch=platform.machine().lower()
        self.tools={}; self.features=[]; self.memory_gb=0.0; self.omp_cflags=[]; self.omp_ldflags=[]; self.llvmlite_version="missing"; self.cflags=""
    def _probe_omp(self,cc):
        if not cc or not shutil.which(cc): return [],[]
        cands=([(["-Xpreprocessor","-fopenmp"],["-lomp"]),(["-fopenmp"],["-fopenmp"])] if self.os_name=="darwin" else [(["-fopenmp"],["-fopenmp"])])
        with tempfile.TemporaryDirectory() as tmp:
            src=os.path.join(tmp,"t.c"); out=os.path.join(tmp,"t")
            with open(src,"w") as f: f.write("#include <omp.h>\nint main(void){return omp_get_max_threads()<1;}\n")
            for cf,lf in cands:
                try:
                    if subprocess.run([cc,*cf,src,*lf,"-o",out],capture_output=True).returncode==0: return cf,lf
                except (FileNotFoundError,OSError): continue
        return [],[]
    def inspect(self):
        for t in ("cc","gcc","clang","ar","make"): self.tools[t]=shutil.which(t) is not None
        try: import llvmlite; self.llvmlite_version=getattr(llvmlite,"__version__","unknown")
        except Exception: self.llvmlite_version="MISSING"
        if self.os_name=="linux":
            try:
                info=open("/proc/cpuinfo").read().lower()
                for tag,lab in (("avx512","AVX-512"),("avx2","AVX2"),("asimd","NEON")):
                    if tag in info and lab not in self.features: self.features.append(lab)
            except OSError: pass
            try:
                for line in open("/proc/meminfo"):
                    if line.startswith("MemTotal:"): self.memory_gb=round(int(line.split()[1])/(1024**2),2); break
            except OSError: pass
        elif self.os_name=="darwin":
            try: self.memory_gb=round(int(subprocess.check_output(["sysctl","-n","hw.memsize"]).strip())/(1024**3),2)
            except Exception: pass
            if self.arch in ("arm64","aarch64"): self.features.append("NEON")
        cc=os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang") or "cc"
        self.omp_cflags,self.omp_ldflags=self._probe_omp(cc)
        base="-O3 -fPIC -std=c11 -Wall -Wextra -Wno-unknown-pragmas -fno-math-errno -pthread"
        if os.environ.get("AXON_NATIVE","0")=="1": base+=" -march=native"
        self.cflags=(base+" "+" ".join(self.omp_cflags)).strip(); return self
    def report(self):
        return "\n".join([f"OS       : {self.os_name} / {self.arch}",f"RAM      : {self.memory_gb} GB",
            f"SIMD     : {', '.join(self.features) or 'generic'}",f"llvmlite : {self.llvmlite_version}",
            "Tools    : "+", ".join(f"{k}={'ok' if v else 'MISSING'}" for k,v in self.tools.items()),
            f"OpenMP   : {' '.join(self.omp_cflags) or 'disabled'}",f"CFLAGS   : {self.cflags}"])

def _half_to_f32(h):
    sign=(h>>15)&1; exp=(h>>10)&0x1F; mant=h&0x3FF
    if exp==0:
        if mant==0: return sign<<31
        e=-1
        while not (mant&0x400): mant<<=1; e-=1
        return (sign<<31)|((127-15+e+1)<<23)|((mant&0x3FF)<<13)
    return (sign<<31)|(0x7F800000 if exp==0x1F else ((exp-15+127)<<23))|(mant<<13)

class Safetensors:
    MAX_HEADER=64*1024*1024
    @staticmethod
    def load(path):
        big=(sys.byteorder=="big")
        with open(path,"rb") as fh:
            head=fh.read(8)
            if len(head)!=8: raise ValueError("truncated header")
            (hlen,)=struct.unpack("<Q",head)
            if hlen==0 or hlen>Safetensors.MAX_HEADER: raise ValueError("bad header length")
            raw=fh.read(hlen)
            if len(raw)!=hlen: raise ValueError("truncated header")
            header=json.loads(raw.decode()); start=8+hlen; fh.seek(0,2); size=fh.tell(); out={}
            for name,meta in header.items():
                if name=="__metadata__": continue
                dt=meta["dtype"]; shape=list(meta["shape"]); beg,end=meta["data_offsets"]
                if not (0<=beg<=end) or start+end>size: raise ValueError(f"{name}: offsets out of bounds")
                numel=1
                for d in shape:
                    if d<0: raise ValueError("negative dim")
                    numel*=d
                fh.seek(start+beg); raw=fh.read(end-beg)
                if dt=="F32":
                    if len(raw)!=numel*4: raise ValueError("F32 size")
                    vals=array("f"); vals.frombytes(raw);
                    if big: vals.byteswap()
                elif dt in ("F16","BF16"):
                    if len(raw)!=numel*2: raise ValueError(f"{dt} size")
                    h=array("H"); h.frombytes(raw)
                    if big: h.byteswap()
                    vals=array("f"); vals.frombytes(b"".join(struct.pack("@I",_half_to_f32(x) if dt=="F16" else (x<<16)) for x in h))
                else: raise ValueError(f"unsupported dtype {dt}")
                out[name]={"shape":shape,"data":vals}
            return out

def _fmt(x):
    v=float(x)
    if not math.isfinite(v): raise AxonError(f"non-finite weight {v}")
    return repr(v)
def _flat(m): return "["+", ".join(_fmt(x) for x in m)+"]"

class Converter:
    @staticmethod
    def generate(model):
        name=model.get("name","model")
        if not name.isidentifier(): raise AxonError(f"invalid name {name}")
        in_shape=list(model.get("input_shape",[1,4])); layers=model.get("layers",[]); prev=list(in_shape); body=[]; var="x"
        for i,l in enumerate(layers):
            lt=l["type"]; out=f"v{i}"
            if lt=="Linear":
                w=l["weights"]
                if not w or not w[0]: raise AxonError("empty weights")
                rows,cols=len(w),len(w[0])
                if any(len(r)!=cols for r in w) or rows!=prev[1]: raise AxonError(f"layer {i}: shape mismatch")
                bias=l.get("bias") or [0.0]*cols
                if len(bias)!=cols: raise AxonError("bias size")
                body.append(f"    let w{i}: f32[{rows}, {cols}] = {_flat([x for r in w for x in r])} as f32[{rows}, {cols}];")
                body.append(f"    let b{i}: f32[{cols}] = {_flat(bias)} as f32[{cols}];")
                nxt=[prev[0],cols]; body.append(f"    let {out}: f32[{nxt[0]}, {nxt[1]}] = add(matmul({var}, w{i}), b{i});"); prev=nxt
            elif lt in ("ReLU","GELU","SiLU","Sigmoid","Softmax"): body.append(f"    let {out}: f32[{prev[0]}, {prev[1]}] = {lt.lower()}({var});")
            elif lt=="LayerNorm":
                g=l.get("gamma") or [1.0]*prev[1]; b=l.get("beta") or [0.0]*prev[1]
                body.append(f"    let g{i}: f32[{prev[1]}] = {_flat(g)} as f32[{prev[1]}];")
                body.append(f"    let be{i}: f32[{prev[1]}] = {_flat(b)} as f32[{prev[1]}];")
                body.append(f"    let {out}: f32[{prev[0]}, {prev[1]}] = layernorm({var}, g{i}, be{i});")
            else: raise AxonError(f"unsupported layer {lt}")
            var=out
        return "\n".join([f"fn {name}(x: f32[{in_shape[0]}, {in_shape[1]}]) -> f32[{prev[0]}, {prev[1]}] {{",*body,f"    return {var};","}",""])
    @staticmethod
    def _find_prefix(t):
        for p in ("l","layers.","model.layers."):
            if f"{p}0.weight" in t: return p
        return None
    @staticmethod
    def from_safetensors(path,transpose=False):
        t=Safetensors.load(path); prefix=Converter._find_prefix(t)
        if prefix is None: raise AxonError("no l{i}.weight / layers.{i}.weight / model.layers.{i}.weight found")
        layers=[]; in_dim=None; idx=0
        while f"{prefix}{idx}.weight" in t:
            wt=t[f"{prefix}{idx}.weight"]
            if len(wt["shape"])!=2: raise AxonError("weight must be 2-D")
            r,c=wt["shape"]; data=wt["data"]
            if transpose: w=[[data[i*c+j] for i in range(r)] for j in range(c)]; k,n=c,r
            else: w=[list(data[i*c:(i+1)*c]) for i in range(r)]; k,n=r,c
            if in_dim is None: in_dim=k
            elif in_dim!=k: raise AxonError("K mismatch")
            bt=t.get(f"{prefix}{idx}.bias"); bias=list(bt["data"]) if bt else [0.0]*n
            layers.append({"type":"Linear","weights":w,"bias":bias})
            act=t.get(f"{prefix}{idx}.act")
            if act is not None:
                code=int(act["data"][0]) if len(act["data"]) else 0
                layers.append({"type":{0:"ReLU",1:"GELU",2:"Softmax",3:"SiLU",4:"Sigmoid"}.get(code,"ReLU")})
            in_dim=n; idx+=1
        return {"name":"safetensors_model","input_shape":[1,len(layers[0]["weights"])],"layers":layers}

EVOLVE_SRC=r"""#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(void){
    const int N=%(N)d,T=%(TILE)d,PAR=%(PAR_MIN)d;
    float *A=malloc((size_t)N*N*4),*B=malloc((size_t)N*N*4),*C=calloc((size_t)N*N,4);
    if(!A||!B||!C) return 2;
    for(int i=0;i<N*N;++i){A[i]=1.0f;B[i]=0.5f;}
    const int use_par=(N*N*N)>=PAR; double best=1e30;
    for(int rep=0;rep<%(REPS)d;++rep){
        struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
        #pragma omp parallel for collapse(2) schedule(static) if(use_par)
        for(int ii=0;ii<N;ii+=T)for(int jj=0;jj<N;jj+=T){
            int ie=ii+T<N?ii+T:N, je=jj+T<N?jj+T:N, wj=je-jj;
            for(int kk=0;kk<N;kk+=T){
                int ke=kk+T<N?kk+T:N, i=ii;
                for(;i+4<=ie;i+=4){
                    float *c0=C+(i+0)*N+jj,*c1=C+(i+1)*N+jj,*c2=C+(i+2)*N+jj,*c3=C+(i+3)*N+jj;
                    for(int k=kk;k<ke;++k){
                        float a0=A[(i+0)*N+k],a1=A[(i+1)*N+k],a2=A[(i+2)*N+k],a3=A[(i+3)*N+k]; const float *bk=B+k*N+jj;
                        #pragma omp simd
                        for(int j=0;j<wj;++j){float bj=bk[j];c0[j]+=a0*bj;c1[j]+=a1*bj;c2[j]+=a2*bj;c3[j]+=a3*bj;}
                    }
                }
                for(;i<ie;++i){ float *ci=C+i*N+jj;
                    for(int k=kk;k<ke;++k){ float aik=A[i*N+k]; const float *bk=B+k*N+jj;
                        #pragma omp simd
                        for(int j=0;j<wj;++j)ci[j]+=aik*bk[j];
                    }
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6;
        if(ms<best)best=ms;
    }
    printf("%%.6f\n",best+C[0]*0.0); free(A);free(B);free(C);return 0;
}
"""
class Evolver:
    @staticmethod
    def _flags():
        insp=Inspector.shared(); f=["-O3","-std=c11","-fno-math-errno","-pthread",*insp.omp_cflags]
        if os.environ.get("AXON_NATIVE","0")=="1" and platform.machine().lower() in ("x86_64","amd64","aarch64","arm64"): f.append("-march=native")
        return f
    @staticmethod
    def _eval(genome,tmp,n,reps):
        src=os.path.join(tmp,"k.c"); exe=os.path.join(tmp,"k")
        with open(src,"w") as f: f.write(EVOLVE_SRC%{"N":n,"TILE":genome["tile"],"PAR_MIN":genome["par_min"],"REPS":reps})
        subprocess.run([os.environ.get("CC") or shutil.which("cc") or "cc",*Evolver._flags(),src,"-o",exe],check=True,capture_output=True)
        return float(subprocess.check_output([exe],text=True).strip())
    @staticmethod
    def evolve(generations=3,pop_size=6,mat_dim=256,seed=1234,write_header=True):
        rng=random.Random(seed); tiles=[16,32,64,128]; pars=[16384,65536,262144]
        pop=[{"tile":t,"par_min":p} for t in tiles for p in pars]; rng.shuffle(pop); pop=pop[:max(2,pop_size)]
        best,best_ms=pop[0],float("inf")
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(max(1,generations)):
                scored=sorted((Evolver._eval(g,tmp,mat_dim,3),g) for g in pop)
                if scored[0][0]<best_ms: best_ms,best=scored[0]
                elites=[g for _,g in scored[:2]]; nxt=list(elites)
                while len(nxt)<len(pop):
                    p=rng.choice(elites); c={"tile":rng.choice(tiles) if rng.random()<0.5 else p["tile"],"par_min":rng.choice(pars) if rng.random()<0.5 else p["par_min"]}
                    nxt.append(c if c not in nxt else {"tile":rng.choice(tiles),"par_min":rng.choice(pars)})
                pop=nxt
        written_to=None
        if write_header:
            header=f"#define AXON_TILE {best['tile']}\n#define AXON_PAR_MIN {best['par_min']}\n"
            target=runtime_dir()/"axon_tuning.h"
            try: target.write_text(header); written_to=str(target)
            except (PermissionError,OSError):
                cache=Path(os.environ.get("AXON_CACHE_DIR",str(Path.home()/".cache"/"axon"))); cache.mkdir(parents=True,exist_ok=True)
                target=cache/"axon_tuning.h"; target.write_text(header); written_to=str(target)
        return {"best_tile":best["tile"],"best_par_min":best["par_min"],"time_ms":round(best_ms,4),"header":written_to}
'''

FILES["axon/cli.py"] = r'''import argparse, ctypes, json, os, shutil, statistics, subprocess, sys, time
from .backend import LLVMBackend, Verilog, Baker
from .blob import BlobPacker, payload_bytes
from .frontend import Lexer, Parser, Semantic
from .ir import IRBuilder, OpKind
from .passes import Pipeline
from .runtime_paths import ensure_runtime
from .tools import Inspector, Converter, Evolver
from .errors import AxonError

def load_graph(path,ternary=False,no_fusion=False,no_quant=False,brain=False):
    with open(path,encoding="utf-8") as fh: src=fh.read()
    ast=Parser(Lexer(src,path).tokenize(),path).parse(); Semantic(path).analyze(ast)
    return Pipeline(fusion=not no_fusion,quant=not no_quant,ternary=ternary,brain=brain).run(IRBuilder().build(ast))

def _bind():
    import llvmlite.binding as llvm
    rt=ensure_runtime(); llvm.load_library_permanently(str(rt.shared_lib)); lib=ctypes.CDLL(str(rt.shared_lib))
    lib.axon_arena_init.argtypes=[ctypes.c_size_t]; lib.axon_arena_reset.argtypes=[]; lib.axon_arena_destroy.argtypes=[]
    lib.axon_arena_used.argtypes=[]; lib.axon_arena_used.restype=ctypes.c_size_t
    lib.axon_live_allocs.argtypes=[]; lib.axon_live_allocs.restype=ctypes.c_int64
    lib.axon_set_weights_blob.argtypes=[ctypes.c_void_p,ctypes.c_int64]
    lib.axon_tensor_alloc.argtypes=[ctypes.c_int64,ctypes.POINTER(ctypes.c_int64),ctypes.c_int32]; lib.axon_tensor_alloc.restype=ctypes.c_void_p
    lib.axon_tensor_alloc_persistent.argtypes=[ctypes.c_int64,ctypes.POINTER(ctypes.c_int64),ctypes.c_int32]; lib.axon_tensor_alloc_persistent.restype=ctypes.c_void_p
    lib.axon_tensor_free.argtypes=[ctypes.c_void_p]
    lib.axon_tensor_fill_f32.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.c_int64]; lib.axon_tensor_fill_f32.restype=ctypes.c_int
    lib.axon_print_tensor.argtypes=[ctypes.c_void_p]; return llvm,lib

def cmd_doctor(args):
    print("=== AXON SYSTEM INSPECTOR ==="); print(Inspector.shared().report())
    try: rt=ensure_runtime(verbose=True); print(f"Runtime  : ok ({rt.dir})")
    except AxonError as e: print(f"Runtime  : FAILED\n{e}"); return 1
    return 0

def cmd_inspect(args):
    g=load_graph(args.source,args.ternary,args.no_fusion,args.no_quant,args.brain)
    raw=sum(len(b.data)*4 for b in g.weights.values()); packed=len(BlobPacker.pack(g.weights))
    rt_bytes=sum(payload_bytes(b) for b in g.weights.values())
    fused=sum(1 for n in g.nodes if n.op.name.startswith("FUSED")); neuro=sum(1 for n in g.nodes if n.op==OpKind.NEURO_MATMUL)
    q={}; [q.__setitem__(b.quant,q.get(b.quant,0)+1) for b in g.weights.values()]
    print("=== AXON MODEL X-RAY ==="); print(f"nodes        : {len(g.nodes)} (fused {fused}, neuro_routed {neuro})"); print(f"weights      : {len(g.weights)} {q}")
    print(f"f32 bytes    : {raw}"); print(f"axn3 blob    : {packed}"+(f" ({(1-packed/raw)*100:.1f}% smaller)" if raw else ""))
    print(f"zero-copy    : {rt_bytes} bytes payload (mmap / rodata view, 0 heap copy)"); return 0

def cmd_build(args):
    g=load_graph(args.source,args.ternary,args.no_fusion,args.no_quant,args.brain)
    if args.emit_silicon:
        out=args.output if args.output.endswith(".v") else args.output+".v"; Verilog.compile(g,out); print(f"[ok] RTL: {out}"); return 0
    ensure_runtime(native=args.native); obj=args.output+".gen.o"
    try:
        LLVMBackend(target_triple=args.target).compile(g,obj)
        Baker(target=args.target,native=args.native).bake([obj],g.weights,args.output,emit_lib=args.emit_lib,external_weights=args.external_weights)
    finally:
        if os.path.exists(obj): os.remove(obj)
    print(f"[ok] {'library' if args.emit_lib else ('binary + mmap .axn3' if args.external_weights else 'binary')}: {args.output}"); return 0

def cmd_jit(args):
    import llvmlite.binding as llvm
    g=load_graph(args.source,args.ternary,args.no_fusion,args.no_quant,args.brain); _,lib=_bind()
    node=g.get(g.inputs[0]); shape=tuple(node.shape); numel=1
    for d in shape: numel*=d
    cshape=(ctypes.c_int64*len(shape))(*shape); seed=[1.0,0.5,-1.0,2.0]
    vals=(ctypes.c_float*numel)(*[seed[i%4] for i in range(numel)])
    inp=lib.axon_tensor_alloc_persistent(len(shape),cshape,0)
    if not inp: print("alloc failed",file=sys.stderr); return 1
    lib.axon_tensor_fill_f32(ctypes.c_void_p(inp),vals,numel); lib.axon_arena_init(64*1024*1024)
    blob=BlobPacker.pack(g.weights); buf=ctypes.create_string_buffer(blob,len(blob))
    lib.axon_set_weights_blob(ctypes.cast(buf,ctypes.c_void_p),len(blob))
    module=LLVMBackend(module_name="axon_jit").parse_for_jit(g); tm=llvm.Target.from_default_triple().create_target_machine(opt=3)
    with llvm.create_mcjit_compiler(module,tm) as engine:
        engine.finalize_object()
        fwd=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p))(engine.get_function_address("axon_forward"))
        shutdown=ctypes.CFUNCTYPE(None)(engine.get_function_address("axon_shutdown"))
        flops_fn=ctypes.CFUNCTYPE(ctypes.c_int64)(engine.get_function_address("axon_get_total_flops"))
        arr=(ctypes.c_void_p*1)(inp)
        if not fwd(arr): print("jit failed",file=sys.stderr); return 1
        samples=[]; out=None
        for _ in range(max(1,args.iters)):
            lib.axon_arena_reset(); t0=time.perf_counter_ns(); out=fwd(arr); samples.append((time.perf_counter_ns()-t0)/1000.0)
        if not out: print("jit failed",file=sys.stderr); return 1
        us=statistics.median(samples); print("[axon mcjit] result:"); lib.axon_print_tensor(ctypes.c_void_p(out))
        fl=flops_fn(); print(f"[mcjit] median {us:.2f} us | est flops {fl} | ~{fl/(us*1e3):.4f} GFLOP/s (estimate)"); shutdown()
    lib.axon_tensor_free(ctypes.c_void_p(inp)); lib.axon_arena_destroy(); print(f"[mcjit] live heap tensors: {lib.axon_live_allocs()}"); return 0

def cmd_convert(args):
    if args.weights.endswith(".safetensors"):
        if not args.transpose and not os.environ.get("AXON_ALLOW_NO_TRANSPOSE"):
            print("[convert] warning: --transpose not set; PyTorch Linear weights are [out,in]",file=sys.stderr)
        model=Converter.from_safetensors(args.weights,args.transpose)
    else:
        with open(args.weights,encoding="utf-8") as fh: model=json.load(fh)
    src=Converter.generate(model); ax=args.output+".axon"
    with open(ax,"w",encoding="utf-8") as fh: fh.write(src)
    try:
        g=load_graph(ax,args.ternary,brain=args.brain); ensure_runtime(native=args.native); obj=args.output+".gen.o"
        try:
            LLVMBackend(target_triple=args.target).compile(g,obj)
            Baker(target=args.target,native=args.native).bake([obj],g.weights,args.output,external_weights=args.external_weights)
        finally:
            if os.path.exists(obj): os.remove(obj)
    finally:
        if not args.keep_source and os.path.exists(ax): os.remove(ax)
    print(f"[ok] converted: {args.output}"); return 0

def cmd_verify(args):
    _,lib=_bind(); n=args.dim
    lib.axon_bench_matmul.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_float)]; lib.axon_bench_matmul.restype=ctypes.c_int
    out=(ctypes.c_float*(n*n))(); ms=ctypes.c_double(0.0); lib.axon_bench_matmul(n,ctypes.byref(ms),out); samples=[]
    for _ in range(3):
        rc=lib.axon_bench_matmul(n,ctypes.byref(ms),out)
        if rc!=0: print(f"kernel error {rc}",file=sys.stderr); return 1
        samples.append(ms.value)
    c_ms=statistics.median(samples); t0=time.perf_counter()
    py=[[sum(((i%10)*0.01)*((j%10)*0.01) for k in range(n)) for j in range(n)] for i in range(n)]
    py_ms=(time.perf_counter()-t0)*1000.0; max_err=max(abs(out[i*n+j]-py[i][j]) for i in range(n) for j in range(n)); tol=1e-3*max(1.0,n)
    print(f"=== AXON VERIFY ({n}x{n}) ==="); print(f"python: {py_ms:.2f} ms | axon C: {c_ms:.3f} ms | speedup: {py_ms/max(c_ms,1e-6):.1f}x")
    ok=max_err<=tol; print(f"max abs error: {max_err:.3e} (tol {tol:.3e}) -> {'PASS' if ok else 'FAIL'}"); return 0 if ok else 1

def cmd_lsh(args): _,lib=_bind(); lib.axon_lsh_benchmark.argtypes=[ctypes.c_int]; lib.axon_lsh_benchmark(args.dim); return 0
def cmd_evolve(args):
    print(f"[evolve] tuning {args.dim}x{args.dim} ..."); best=Evolver.evolve(generations=args.generations,mat_dim=args.dim)
    print(f"[evolve] best tile={best['best_tile']} par_min={best['best_par_min']} latency={best['time_ms']} ms -> {best['header']}")
    ensure_runtime(verbose=True); return 0
def cmd_run(args):
    exe=args.binary if os.path.sep in args.binary else os.path.join(".",args.binary); cmd=[exe]
    if args.input: cmd+=["--input",args.input]
    if args.iters: cmd+=["--iters",str(args.iters)]
    return subprocess.run(cmd).returncode
def cmd_selfcheck(args):
    rc=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_*.py","-v"]).returncode
    if rc!=0: return rc
    if shutil.which("iverilog") and shutil.which("vvp"):
        tmp_files=["selfcheck_chip.v","selfcheck_chip_tb.v","selfcheck_chip_weights.hex","selfcheck_chip.sim"]
        try:
            g=load_graph("examples/transformer_block.axon"); Verilog.compile(g,"selfcheck_chip.v")
            r=subprocess.run(["iverilog","-o","selfcheck_chip.sim","selfcheck_chip.v","selfcheck_chip_tb.v"],capture_output=True,text=True)
            if r.returncode!=0: print(r.stderr); return 1
            r=subprocess.run(["vvp","selfcheck_chip.sim"],capture_output=True,text=True); print(r.stdout.strip())
            if "PASS" not in r.stdout: return 1
        finally:
            for tf in tmp_files:
                if os.path.exists(tf): os.remove(tf)
    else: print("[selfcheck] iverilog/vvp not found; RTL simulation skipped")
    print("[selfcheck] PASS"); return 0

def build_parser():
    p=argparse.ArgumentParser(prog="axon"); sp=p.add_subparsers(dest="command",required=True)
    sp.add_parser("doctor").set_defaults(func=cmd_doctor); sp.add_parser("selfcheck").set_defaults(func=cmd_selfcheck)
    pv=sp.add_parser("verify"); pv.add_argument("--dim",type=int,default=128); pv.set_defaults(func=cmd_verify)
    def common(pp):
        pp.add_argument("source"); pp.add_argument("--ternary",action="store_true"); pp.add_argument("--no-fusion",action="store_true")
        pp.add_argument("--no-quant",action="store_true"); pp.add_argument("--brain",action="store_true")
    pi=sp.add_parser("inspect"); common(pi); pi.set_defaults(func=cmd_inspect)
    pj=sp.add_parser("jit"); common(pj); pj.add_argument("--iters",type=int,default=5); pj.set_defaults(func=cmd_jit)
    pb=sp.add_parser("build"); common(pb); pb.add_argument("-o","--output",default="model.bin"); pb.add_argument("--target",default="")
    pb.add_argument("--native",action="store_true"); pb.add_argument("--emit-lib",action="store_true"); pb.add_argument("--emit-silicon",action="store_true")
    pb.add_argument("--external-weights",action="store_true"); pb.set_defaults(func=cmd_build)
    pc=sp.add_parser("convert"); pc.add_argument("weights"); pc.add_argument("-o","--output",default="model.bin"); pc.add_argument("--target",default="")
    pc.add_argument("--native",action="store_true"); pc.add_argument("--ternary",action="store_true"); pc.add_argument("--brain",action="store_true")
    pc.add_argument("--external-weights",action="store_true"); pc.add_argument("--transpose",action="store_true"); pc.add_argument("--keep-source",action="store_true"); pc.set_defaults(func=cmd_convert)
    pl=sp.add_parser("lsh-bench"); pl.add_argument("--dim",type=int,default=512); pl.set_defaults(func=cmd_lsh)
    pe=sp.add_parser("evolve"); pe.add_argument("--dim",type=int,default=256); pe.add_argument("--generations",type=int,default=3); pe.set_defaults(func=cmd_evolve)
    pr=sp.add_parser("run"); pr.add_argument("binary"); pr.add_argument("--input",default=""); pr.add_argument("--iters",type=int,default=1); pr.set_defaults(func=cmd_run)
    return p

def main(argv=None):
    args=build_parser().parse_args(argv)
    try: rc=args.func(args)
    except AxonError as e: print(str(e),file=sys.stderr); return 2
    except FileNotFoundError as e: print(f"file not found: {e}",file=sys.stderr); return 2
    except KeyboardInterrupt: return 130
    return int(rc or 0)
if __name__=="__main__": sys.exit(main())
'''

FILES["axon/runtime/__init__.py"] = r'''# Axon C runtime sources.'''

FILES["axon/runtime/axon_runtime.h"] = r'''#ifndef AXON_RUNTIME_H
#define AXON_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
#define AXON_MAX_DIMS 4
#define AXON_DTYPE_F32 0
#define AXON_DTYPE_FP4 2
#define AXON_DTYPE_TERNARY 4
#define AXON_OK 0
#define AXON_ERR_ARG (-1)
#define AXON_ERR_SHAPE (-2)
#define AXON_ERR_DTYPE (-3)
#define AXON_ERR_NOMEM (-4)
#define AXON_ERR_FORMAT (-5)
typedef struct AxonTensor {
    void *data; int64_t ndim; int64_t shape[AXON_MAX_DIMS]; int64_t strides[AXON_MAX_DIMS];
    int32_t dtype; int32_t reserved; float scale; bool owns_data; bool in_arena;
} AxonTensor;
void axon_arena_init(size_t cap); void axon_arena_reset(void); void axon_arena_destroy(void);
bool axon_arena_active(void); size_t axon_arena_used(void); size_t axon_arena_capacity(void);
AxonTensor *axon_tensor_alloc(int64_t ndim, const int64_t *shape, int32_t dtype);
AxonTensor *axon_tensor_alloc_persistent(int64_t ndim, const int64_t *shape, int32_t dtype);
void axon_tensor_free(AxonTensor *t); int64_t axon_tensor_numel(const AxonTensor *t);
int axon_tensor_fill_f32(AxonTensor *t, const float *v, int64_t count);
void axon_print_tensor(const AxonTensor *t); void axon_print_tensor_n(const AxonTensor *t, int64_t lim);
int64_t axon_live_allocs(void); void axon_set_weights_blob(const void *blob, int64_t size);
int axon_load_weights_file(const char *path); void axon_unload_weights_file(void);
AxonTensor *axon_weight_tensor(const char *name); void axon_weights_get(const void **p, int64_t *n);
void axon_rt_lock(void); void axon_rt_unlock(void);
int axon_matmul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_neuro_matmul(const AxonTensor *X, const AxonTensor *W, const AxonTensor *S, AxonTensor *C);
int axon_add(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_sub(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_mul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_relu(const AxonTensor *A, AxonTensor *C); int axon_gelu(const AxonTensor *A, AxonTensor *C);
int axon_silu(const AxonTensor *A, AxonTensor *C); int axon_sigmoid(const AxonTensor *A, AxonTensor *C);
int axon_softmax(const AxonTensor *A, AxonTensor *C); int axon_transpose(const AxonTensor *A, AxonTensor *C);
int axon_layer_norm(const AxonTensor *X, const AxonTensor *G, const AxonTensor *B, AxonTensor *C);
int axon_equilibrium(const AxonTensor *X, const AxonTensor *W, const AxonTensor *B, AxonTensor *C);
int axon_mm_relu(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_mm_gelu(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_mm_silu(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
int axon_mm_bias(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
int axon_mm_bias_relu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
int axon_mm_bias_gelu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
int axon_mm_bias_silu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
int axon_bench_matmul(int n, double *out_ms, float *out_c); void axon_lsh_benchmark(int dim);
int axon_init(void); void axon_shutdown(void); AxonTensor *axon_forward(AxonTensor **inputs);
int64_t axon_get_input_ndim(void); void axon_get_input_shape(int64_t *out);
int64_t axon_get_input_numel(void); int64_t axon_get_total_flops(void); int64_t axon_output_is_temp(void);
#ifdef __cplusplus
}
#endif
#endif
'''

FILES["axon/runtime/axon_runtime.c"] = r'''#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "axon_runtime.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__unix__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif
#if defined(__has_include)
#  if __has_include("axon_tuning.h")
#    include "axon_tuning.h"
#  endif
#endif
#ifndef AXON_TILE
#define AXON_TILE 64
#endif
#if AXON_TILE < 1
#error "AXON_TILE must be >= 1"
#endif
#if AXON_TILE > 64
#define AXON_BT 64
#else
#define AXON_BT AXON_TILE
#endif
#if AXON_BT < 1
#error "AXON_BT must be >= 1"
#endif
#ifndef AXON_PAR_MIN
#define AXON_PAR_MIN 65536
#endif
#define AXON_ALIGN 64
#define SQRT_2_OVER_PI 0.7978845608028654f
#define GELU_C 0.044715f
#define LN_EPS 1e-5f
#define REQ(c,e) do{ if(!(c)) return (e); }while(0)

static _Thread_local uint8_t *g_base=NULL;
static _Thread_local size_t g_cap=0,g_off=0;
static atomic_int_fast64_t g_live=0;
static void live_inc(void){ atomic_fetch_add_explicit(&g_live,1,memory_order_relaxed); }
static void live_dec(void){ atomic_fetch_sub_explicit(&g_live,1,memory_order_relaxed); }
void axon_arena_init(size_t cap){ if(g_base||cap==0)return; void*p=NULL; if(posix_memalign(&p,AXON_ALIGN,cap)==0){g_base=(uint8_t*)p;g_cap=cap;g_off=0;} }
static void* arena_alloc(size_t n){ if(!g_base)return NULL; size_t a=(n+(AXON_ALIGN-1))&~((size_t)(AXON_ALIGN-1)); if(a<n||g_off>g_cap||a>g_cap-g_off)return NULL; void*p=g_base+g_off; g_off+=a; return p; }
void axon_arena_reset(void){g_off=0;}
void axon_arena_destroy(void){free(g_base);g_base=NULL;g_cap=0;g_off=0;}
bool axon_arena_active(void){return g_base!=NULL;}
size_t axon_arena_used(void){return g_off;}
size_t axon_arena_capacity(void){return g_cap;}
int64_t axon_live_allocs(void){return (int64_t)atomic_load_explicit(&g_live,memory_order_relaxed);}

static size_t elemsz(int32_t dt,int64_t total){ if(dt==AXON_DTYPE_TERNARY)return (size_t)((total+3)/4); if(dt==AXON_DTYPE_FP4)return (size_t)((total+1)/2); return (size_t)total*sizeof(float); }
static AxonTensor* tensor_new(int64_t ndim,const int64_t*shape,int32_t dtype,int persistent){
    if(ndim<1||ndim>AXON_MAX_DIMS||!shape)return NULL;
    if(dtype!=AXON_DTYPE_F32&&dtype!=AXON_DTYPE_TERNARY&&dtype!=AXON_DTYPE_FP4)return NULL;
    int64_t total=1;
    for(int64_t i=0;i<ndim;++i){ if(shape[i]<=0||shape[i]>INT64_MAX/total)return NULL; total*=shape[i]; }
    if((uint64_t)total>(uint64_t)(SIZE_MAX/sizeof(float)))return NULL;
    size_t bytes=elemsz(dtype,total); if(bytes<AXON_ALIGN)bytes=AXON_ALIGN;
    int use_arena=(!persistent&&axon_arena_active());
    AxonTensor*t=use_arena?(AxonTensor*)arena_alloc(sizeof(AxonTensor)):NULL;
    if(!t){ t=(AxonTensor*)malloc(sizeof(AxonTensor)); if(!t)return NULL; use_arena=0; live_inc(); }
    t->ndim=ndim;t->dtype=dtype;t->reserved=0;t->scale=1.0f;t->in_arena=(bool)use_arena;t->owns_data=false;t->data=NULL;
    memset(t->shape,0,sizeof(t->shape));memset(t->strides,0,sizeof(t->strides));
    for(int64_t i=0;i<ndim;++i)t->shape[i]=shape[i];
    int64_t s=1; for(int64_t i=ndim-1;i>=0;--i){t->strides[i]=s;s*=t->shape[i];}
    if(use_arena){ t->data=arena_alloc(bytes); if(t->data)return t; }
    void*d=NULL; if(posix_memalign(&d,AXON_ALIGN,bytes)!=0){ if(!t->in_arena){free(t);live_dec();} return NULL; }
    t->data=d;t->owns_data=true; if(dtype==AXON_DTYPE_F32)memset(d,0,bytes); return t;
}
static AxonTensor* tensor_view_new(int64_t ndim,const int64_t*shape,int32_t dtype,const void*ptr){
    if(ndim<1||ndim>AXON_MAX_DIMS||!shape||!ptr)return NULL;
    int64_t total=1; for(int64_t i=0;i<ndim;++i){ if(shape[i]<=0||shape[i]>INT64_MAX/total)return NULL; total*=shape[i]; }
    AxonTensor*t=(AxonTensor*)malloc(sizeof(AxonTensor)); if(!t)return NULL; live_inc();
    t->ndim=ndim;t->dtype=dtype;t->reserved=0;t->scale=1.0f;t->in_arena=false;t->owns_data=false;t->data=(void*)ptr;
    memset(t->shape,0,sizeof(t->shape));memset(t->strides,0,sizeof(t->strides));
    for(int64_t i=0;i<ndim;++i)t->shape[i]=shape[i];
    int64_t s=1; for(int64_t i=ndim-1;i>=0;--i){t->strides[i]=s;s*=t->shape[i];}
    return t;
}
AxonTensor* axon_tensor_alloc(int64_t ndim,const int64_t*shape,int32_t dtype){return tensor_new(ndim,shape,dtype,0);}
AxonTensor* axon_tensor_alloc_persistent(int64_t ndim,const int64_t*shape,int32_t dtype){return tensor_new(ndim,shape,dtype,1);}
void axon_tensor_free(AxonTensor*t){ if(!t)return; if(t->owns_data)free(t->data); if(!t->in_arena){free(t);live_dec();} }
int64_t axon_tensor_numel(const AxonTensor*t){ if(!t||t->ndim<1)return 0; int64_t n=1; for(int64_t i=0;i<t->ndim;++i)n*=t->shape[i]; return n; }
int axon_tensor_fill_f32(AxonTensor*t,const float*v,int64_t count){ REQ(t&&t->data&&v,AXON_ERR_ARG); REQ(t->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE); int64_t n=axon_tensor_numel(t); int64_t lim=count<n?count:n; memcpy(t->data,v,(size_t)lim*sizeof(float)); return AXON_OK; }

static const float FP4_LUT[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
void axon_print_tensor_n(const AxonTensor*t,int64_t max_lim){
    if(!t){printf("Tensor: NULL\n");return;}
    printf("Tensor(ndim=%ld, shape=[",(long)t->ndim);
    for(int64_t i=0;i<t->ndim;++i)printf("%ld%s",(long)t->shape[i],(i+1<t->ndim)?", ":"");
    printf("], dtype=%d) = [\n",(int)t->dtype);
    int64_t total=axon_tensor_numel(t), lim=(max_lim>0&&max_lim<total)?max_lim:total;
    if(t->dtype==AXON_DTYPE_TERNARY){
        const uint8_t*w=(const uint8_t*)t->data;
        for(int64_t i=0;i<lim;++i){uint8_t c=(uint8_t)((w[i>>2]>>((i&3)*2))&3); printf("  %.9g\n",c==1?t->scale:(c==2?-t->scale:0.0f));}
    } else if(t->dtype==AXON_DTYPE_FP4){
        const uint8_t*p=(const uint8_t*)t->data; uint32_t nb=0,nr=0; memcpy(&nb,p,4); memcpy(&nr,p+4,4);
        const uint8_t*scales=p+8; const uint8_t*packed=scales+nb; const uint8_t*res=packed+(size_t)nb*16u;
        for(int64_t i=0;i<lim;++i){
            uint32_t blk=(uint32_t)(i>>5); int off=(int)(i&31); uint8_t byte=packed[(size_t)blk*16u+(size_t)(off>>1)];
            float v=FP4_LUT[(off&1)?((byte>>4)&0x0F):(byte&0x0F)]*ldexpf(1.0f,(int)scales[blk]-127);
            for(uint32_t r=0;r<nr;++r){ uint32_t idx; float diff; memcpy(&idx,res+(size_t)r*8u,4); memcpy(&diff,res+(size_t)r*8u+4,4); if((int64_t)idx==i)v+=diff; }
            printf("  %.9g\n",(double)v);
        }
    } else { const float*d=(const float*)t->data; for(int64_t i=0;i<lim;++i)printf("  %.9g\n",(double)d[i]); }
    if(total>lim)printf("  ... (%ld more)\n",(long)(total-lim));
    printf("]\n");
}
void axon_print_tensor(const AxonTensor*t){ axon_print_tensor_n(t,16); }

static pthread_mutex_t g_mu; static pthread_once_t g_mu_once=PTHREAD_ONCE_INIT;
static void mu_init(void){ pthread_mutexattr_t a; pthread_mutexattr_init(&a); pthread_mutexattr_settype(&a,PTHREAD_MUTEX_RECURSIVE); pthread_mutex_init(&g_mu,&a); pthread_mutexattr_destroy(&a); }
void axon_rt_lock(void){ pthread_once(&g_mu_once,mu_init); pthread_mutex_lock(&g_mu); }
void axon_rt_unlock(void){ pthread_mutex_unlock(&g_mu); }

typedef struct __attribute__((packed)){ uint64_t hash,name_off,data_off,nbytes,dtype,ndim; int64_t shape[4]; } AxonEntry;
typedef struct __attribute__((packed)){ char magic[4]; uint32_t version,count,reserved; uint64_t total_size; } AxonHeader;
static const uint8_t*g_blob=NULL; static uint64_t g_blob_size=0;
static void*g_mmap_ptr=NULL; static size_t g_mmap_size=0;
static int blob_valid(void){
    if(!g_blob||g_blob_size<sizeof(AxonHeader))return 0;
    AxonHeader h; memcpy(&h,g_blob,sizeof(h));
    return memcmp(h.magic,"AXN3",4)==0&&h.version==3&&h.total_size==g_blob_size&&sizeof(AxonHeader)+(uint64_t)h.count*sizeof(AxonEntry)<=g_blob_size;
}
void axon_set_weights_blob(const void*blob,int64_t size){ axon_rt_lock(); g_blob=(const uint8_t*)blob; g_blob_size=size>0?(uint64_t)size:0; axon_rt_unlock(); }
void axon_unload_weights_file(void){
    axon_rt_lock();
#if defined(__unix__) || defined(__APPLE__)
    if(g_mmap_ptr){ munmap(g_mmap_ptr,g_mmap_size); g_mmap_ptr=NULL; g_mmap_size=0; g_blob=NULL; g_blob_size=0; }
#endif
    axon_rt_unlock();
}
int axon_load_weights_file(const char*path){
#if defined(__unix__) || defined(__APPLE__)
    if(!path)return AXON_ERR_ARG; axon_unload_weights_file();
    int fd=open(path,O_RDONLY); if(fd<0)return AXON_ERR_ARG;
    struct stat st; if(fstat(fd,&st)!=0||st.st_size<(off_t)sizeof(AxonHeader)){close(fd);return AXON_ERR_FORMAT;}
    size_t sz=(size_t)st.st_size; void*p=mmap(NULL,sz,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(p==MAP_FAILED)return AXON_ERR_NOMEM;
#ifdef MADV_RANDOM
    madvise(p,sz,MADV_RANDOM);
#endif
    axon_rt_lock(); g_mmap_ptr=p; g_mmap_size=sz; g_blob=(const uint8_t*)p; g_blob_size=(uint64_t)sz;
    int ok=blob_valid(); axon_rt_unlock();
    if(!ok){ axon_unload_weights_file(); return AXON_ERR_FORMAT; }
    return AXON_OK;
#else
    (void)path; return AXON_ERR_ARG;
#endif
}
static uint64_t fnv1a(const char*s){ uint64_t h=14695981039346656037ULL; while(*s){h^=(uint8_t)(*s++);h*=1099511628211ULL;} return h; }
static const AxonEntry* find_entry(const char*name){
    if(!blob_valid())return NULL;
    AxonHeader h; memcpy(&h,g_blob,sizeof(h));
    uint64_t names_base=sizeof(AxonHeader)+(uint64_t)h.count*sizeof(AxonEntry);
    if(names_base>g_blob_size)return NULL;
    uint64_t key=fnv1a(name);
    for(uint64_t i=0;i<h.count;++i){
        AxonEntry e; memcpy(&e,g_blob+sizeof(AxonHeader)+i*sizeof(AxonEntry),sizeof(e));
        if(e.hash!=key||e.name_off>=g_blob_size-names_base)continue;
        const char*nm=(const char*)(g_blob+names_base+e.name_off); size_t maxlen=(size_t)(g_blob_size-(names_base+e.name_off));
        if(memchr(nm,0,maxlen)==NULL||strcmp(nm,name)!=0)continue;
        if(e.data_off>g_blob_size||e.nbytes>g_blob_size-e.data_off||e.ndim<1||e.ndim>AXON_MAX_DIMS)continue;
        return (const AxonEntry*)(g_blob+sizeof(AxonHeader)+i*sizeof(AxonEntry));
    }
    return NULL;
}
AxonTensor* axon_weight_tensor(const char*name){
    if(!name)return NULL; const AxonEntry*e=find_entry(name); if(!e)return NULL;
    AxonEntry ent; memcpy(&ent,e,sizeof(ent)); const uint8_t*payload=g_blob+ent.data_off;
    int64_t total=1; for(int64_t i=0;i<ent.ndim;++i){ if(ent.shape[i]<=0)return NULL; total*=ent.shape[i]; }
    if(ent.dtype==AXON_DTYPE_F32){
        if(ent.nbytes!=(uint64_t)total*sizeof(float))return NULL;
        return tensor_view_new(ent.ndim,ent.shape,AXON_DTYPE_F32,payload);
    }
    if(ent.dtype==AXON_DTYPE_TERNARY){
        if(ent.nbytes<8)return NULL; float scale; uint32_t n; memcpy(&scale,payload,4); memcpy(&n,payload+4,4);
        if((int64_t)n!=total||ent.nbytes<8+(uint64_t)((total+3)/4))return NULL;
        AxonTensor*t=tensor_view_new(ent.ndim,ent.shape,AXON_DTYPE_TERNARY,payload+8);
        if(t)t->scale=scale; return t;
    }
    if(ent.dtype==AXON_DTYPE_FP4){
        if(ent.nbytes<8)return NULL; uint32_t nb,nr; memcpy(&nb,payload,4); memcpy(&nr,payload+4,4);
        uint64_t need=8ull+nb+16ull*nb+8ull*nr; if(ent.nbytes<need||(int64_t)nb*32<total)return NULL;
        AxonTensor*t=tensor_view_new(ent.ndim,ent.shape,AXON_DTYPE_FP4,payload); if(!t)return NULL;
        const uint8_t*res=payload+8ull+nb+16ull*nb; int sorted=(ent.ndim==2)?1:0;
        if(sorted&&nr>0){
            int64_t N=ent.shape[1], prev_rj=-1;
            for(uint32_t r=0;r<nr;++r){
                uint32_t idx; memcpy(&idx,res+(size_t)r*8u,4);
                if((int64_t)idx>=total){sorted=0;break;}
                int64_t rj=(int64_t)idx%N; if(rj<prev_rj){sorted=0;break;} prev_rj=rj;
            }
        }
        t->reserved=sorted; return t;
    }
    return NULL;
}

static inline float sigmoid_f(float v){ if(v>=0.f)return 1.f/(1.f+expf(-v)); float e=expf(v); return e/(1.f+e); }
static inline float act_apply(float v,int act){
    if(act==1)return v>0.f?v:0.f;
    if(act==2)return 0.5f*v*(1.f+tanhf(SQRT_2_OVER_PI*(v+GELU_C*v*v*v)));
    if(act==3)return v*sigmoid_f(v);
    if(act==4)return sigmoid_f(v);
    return v;
}
static uint32_t fp4_res_lb(const uint8_t*res,uint32_t nr,int64_t N,int64_t jj){
    uint32_t lo=0,hi=nr;
    while(lo<hi){ uint32_t mid=lo+(hi-lo)/2, midx=0; memcpy(&midx,res+(size_t)mid*8u,4); if(((int64_t)midx%N)<jj)lo=mid+1; else hi=mid; }
    return lo;
}
static int gemm(const AxonTensor*A,const AxonTensor*B,const AxonTensor*bias,int act,AxonTensor*C){
    REQ(A&&B&&C&&A->data&&B->data&&C->data,AXON_ERR_ARG);
    REQ(A->ndim==2&&B->ndim==2&&C->ndim==2,AXON_ERR_SHAPE);
    REQ(A->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(B->dtype==AXON_DTYPE_F32||B->dtype==AXON_DTYPE_TERNARY||B->dtype==AXON_DTYPE_FP4,AXON_ERR_DTYPE);
    const int64_t M=A->shape[0],K=A->shape[1],N=B->shape[1]; REQ(B->shape[0]==K&&C->shape[0]==M&&C->shape[1]==N,AXON_ERR_SHAPE);
    const float*bv=NULL; if(bias){ REQ(bias->data&&bias->ndim==1&&bias->shape[0]==N&&bias->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE); bv=(const float*)bias->data; }
    const float*a=(const float*)A->data; float*c=(float*)C->data; const int par=(M*N*K)>=AXON_PAR_MIN;
    if(B->dtype==AXON_DTYPE_TERNARY||B->dtype==AXON_DTYPE_FP4){
        const int is_fp4=(B->dtype==AXON_DTYPE_FP4);
        const uint8_t*tw=(const uint8_t*)B->data; const float lut[4]={0.f,B->scale,-B->scale,0.f};
        uint32_t nb=0,nr=0; const uint8_t*scales=NULL,*packed=NULL,*res=NULL;
        if(is_fp4){ memcpy(&nb,tw,4); memcpy(&nr,tw+4,4); scales=tw+8; packed=scales+nb; res=packed+(size_t)nb*16u; }
        const int64_t T=AXON_BT; const int sorted=(B->reserved==1);
        #pragma omp parallel for collapse(2) schedule(static) if(par)
        for(int64_t ii=0;ii<M;ii+=T)for(int64_t jj=0;jj<N;jj+=T){
            const int64_t ie=(ii+T<M)?ii+T:M, je=(jj+T<N)?jj+T:N, wj=je-jj;
            for(int64_t i=ii;i<ie;++i)for(int64_t j=jj;j<je;++j)c[i*N+j]=0.f;
            float bt[AXON_BT][AXON_BT];
            for(int64_t kk=0;kk<K;kk+=T){
                const int64_t ke=(kk+T<K)?kk+T:K; uint32_t cur_blk=UINT32_MAX; float cur_sc=1.0f;
                for(int64_t k=kk;k<ke;++k)for(int64_t j=jj;j<je;++j){
                    int64_t idx=k*N+j;
                    if(is_fp4){
                        uint32_t blk=(uint32_t)(idx>>5); if(blk!=cur_blk){cur_blk=blk;cur_sc=ldexpf(1.0f,(int)scales[blk]-127);}
                        int off=(int)(idx&31); uint8_t byte=packed[(size_t)blk*16u+(size_t)(off>>1)];
                        bt[k-kk][j-jj]=FP4_LUT[(off&1)?((byte>>4)&0x0F):(byte&0x0F)]*cur_sc;
                    } else { bt[k-kk][j-jj]=lut[(tw[idx>>2]>>((idx&3)*2))&3]; }
                }
                int64_t i=ii;
                for(;i+4<=ie;i+=4){
                    float*c0=c+(i+0)*N+jj,*c1=c+(i+1)*N+jj,*c2=c+(i+2)*N+jj,*c3=c+(i+3)*N+jj;
                    for(int64_t k=kk;k<ke;++k){
                        const float a0=a[(i+0)*K+k],a1=a[(i+1)*K+k],a2=a[(i+2)*K+k],a3=a[(i+3)*K+k]; const float*brow=bt[k-kk];
                        #pragma omp simd
                        for(int64_t j=0;j<wj;++j){const float bj=brow[j];c0[j]+=a0*bj;c1[j]+=a1*bj;c2[j]+=a2*bj;c3[j]+=a3*bj;}
                    }
                }
                for(;i<ie;++i){ float*ci=c+i*N+jj;
                    for(int64_t k=kk;k<ke;++k){ const float aik=a[i*K+k]; const float*brow=bt[k-kk];
                        #pragma omp simd
                        for(int64_t j=0;j<wj;++j)ci[j]+=aik*brow[j];
                    }
                }
            }
            if(is_fp4&&nr>0){
                if(sorted){
                    for(uint32_t r=fp4_res_lb(res,nr,N,jj);r<nr;++r){
                        uint32_t idx; float diff; memcpy(&idx,res+(size_t)r*8u,4); memcpy(&diff,res+(size_t)r*8u+4,4);
                        int64_t rk=(int64_t)idx/N, rj=(int64_t)idx%N; if(rj>=je)break;
                        for(int64_t i=ii;i<ie;++i)c[i*N+rj]+=a[i*K+rk]*diff;
                    }
                } else {
                    for(uint32_t r=0;r<nr;++r){
                        uint32_t idx; float diff; memcpy(&idx,res+(size_t)r*8u,4); memcpy(&diff,res+(size_t)r*8u+4,4);
                        if((int64_t)idx<K*N){ int64_t rk=(int64_t)idx/N, rj=(int64_t)idx%N;
                            if(rj>=jj&&rj<je)for(int64_t i=ii;i<ie;++i)c[i*N+rj]+=a[i*K+rk]*diff;
                        }
                    }
                }
            }
            for(int64_t i=ii;i<ie;++i){
                #pragma omp simd
                for(int64_t j=jj;j<je;++j){float v=c[i*N+j];if(bv)v+=bv[j];c[i*N+j]=act_apply(v,act);}
            }
        }
        return AXON_OK;
    }
    const float*b=(const float*)B->data;
    #pragma omp parallel for collapse(2) schedule(static) if(par)
    for(int64_t ii=0;ii<M;ii+=AXON_TILE)for(int64_t jj=0;jj<N;jj+=AXON_TILE){
        const int64_t ie=(ii+AXON_TILE<M)?ii+AXON_TILE:M, je=(jj+AXON_TILE<N)?jj+AXON_TILE:N, wj=je-jj;
        for(int64_t i=ii;i<ie;++i)for(int64_t j=jj;j<je;++j)c[i*N+j]=0.f;
        for(int64_t kk=0;kk<K;kk+=AXON_TILE){
            const int64_t ke=(kk+AXON_TILE<K)?kk+AXON_TILE:K; int64_t i=ii;
            for(;i+4<=ie;i+=4){
                float*c0=c+(i+0)*N+jj,*c1=c+(i+1)*N+jj,*c2=c+(i+2)*N+jj,*c3=c+(i+3)*N+jj;
                for(int64_t k=kk;k<ke;++k){
                    const float a0=a[(i+0)*K+k],a1=a[(i+1)*K+k],a2=a[(i+2)*K+k],a3=a[(i+3)*K+k]; const float*bk=b+k*N+jj;
                    #pragma omp simd
                    for(int64_t j=0;j<wj;++j){const float bj=bk[j];c0[j]+=a0*bj;c1[j]+=a1*bj;c2[j]+=a2*bj;c3[j]+=a3*bj;}
                }
            }
            for(;i<ie;++i){ float*ci=c+i*N+jj;
                for(int64_t k=kk;k<ke;++k){ const float aik=a[i*K+k]; const float*bk=b+k*N+jj;
                    #pragma omp simd
                    for(int64_t j=0;j<wj;++j)ci[j]+=aik*bk[j];
                }
            }
        }
        for(int64_t i=ii;i<ie;++i){
            #pragma omp simd
            for(int64_t j=jj;j<je;++j){ float v=c[i*N+j]; if(bv)v+=bv[j]; c[i*N+j]=act_apply(v,act); }
        }
    }
    return AXON_OK;
}
int axon_matmul(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return gemm(A,B,NULL,0,C);}
int axon_mm_relu(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return gemm(A,B,NULL,1,C);}
int axon_mm_gelu(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return gemm(A,B,NULL,2,C);}
int axon_mm_silu(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return gemm(A,B,NULL,3,C);}
int axon_mm_bias(const AxonTensor*A,const AxonTensor*B,const AxonTensor*b,AxonTensor*C){return gemm(A,B,b,0,C);}
int axon_mm_bias_relu(const AxonTensor*A,const AxonTensor*B,const AxonTensor*b,AxonTensor*C){return gemm(A,B,b,1,C);}
int axon_mm_bias_gelu(const AxonTensor*A,const AxonTensor*B,const AxonTensor*b,AxonTensor*C){return gemm(A,B,b,2,C);}
int axon_mm_bias_silu(const AxonTensor*A,const AxonTensor*B,const AxonTensor*b,AxonTensor*C){return gemm(A,B,b,3,C);}

int axon_neuro_matmul(const AxonTensor*X,const AxonTensor*W,const AxonTensor*S,AxonTensor*C){
    REQ(X&&W&&S&&C&&X->data&&W->data&&S->data&&C->data,AXON_ERR_ARG);
    REQ(X->ndim==2&&W->ndim==2&&S->ndim==2&&C->ndim==2,AXON_ERR_SHAPE);
    REQ(X->dtype==AXON_DTYPE_F32&&S->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(W->dtype==AXON_DTYPE_F32||W->dtype==AXON_DTYPE_TERNARY||W->dtype==AXON_DTYPE_FP4,AXON_ERR_DTYPE);
    const int64_t M=X->shape[0],K=X->shape[1],N=W->shape[1],R=S->shape[1];
    REQ(W->shape[0]==K&&S->shape[0]==K&&R>=1&&R<=N&&C->shape[0]==M&&C->shape[1]==N,AXON_ERR_SHAPE);
    const int64_t keep=(R>=2)?(R/2):1;
    const float*x=(const float*)X->data,*s=(const float*)S->data; float*c=(float*)C->data;
    memset(c,0,(size_t)(M*N)*sizeof(float));
    float st_sc[64]; int64_t st_ord[64];
    float*sc=(R<=64)?st_sc:(float*)malloc((size_t)R*sizeof(float));
    int64_t*ord=(R<=64)?st_ord:(int64_t*)malloc((size_t)R*sizeof(int64_t));
    if(!sc||!ord){ if(R>64){free(sc);free(ord);} return AXON_ERR_NOMEM; }
    for(int64_t i=0;i<M;++i){
        for(int64_t r=0;r<R;++r){ float acc=0.f; for(int64_t k=0;k<K;++k)acc+=x[i*K+k]*s[k*R+r]; sc[r]=fabsf(acc); ord[r]=r; }
        for(int64_t t=0;t<keep;++t){
            int64_t best=t;
            for(int64_t u=t+1;u<R;++u)if(sc[ord[u]]>sc[ord[best]]||(sc[ord[u]]==sc[ord[best]]&&ord[u]<ord[best]))best=u;
            int64_t tmp=ord[t]; ord[t]=ord[best]; ord[best]=tmp;
        }
        for(int64_t t=0;t<keep;++t){
            int64_t r=ord[t], c0=(r*N)/R, c1=((r+1)*N)/R;
            if(W->dtype==AXON_DTYPE_F32){
                const float*w=(const float*)W->data;
                for(int64_t k=0;k<K;++k){ const float xik=x[i*K+k];
                    #pragma omp simd
                    for(int64_t j=c0;j<c1;++j)c[i*N+j]+=xik*w[k*N+j];
                }
            } else if(W->dtype==AXON_DTYPE_TERNARY){
                const uint8_t*w=(const uint8_t*)W->data; const float lut[4]={0.f,W->scale,-W->scale,0.f};
                for(int64_t k=0;k<K;++k){ const float xik=x[i*K+k];
                    for(int64_t j=c0;j<c1;++j){ int64_t idx=k*N+j; c[i*N+j]+=xik*lut[(w[idx>>2]>>((idx&3)*2))&3]; }
                }
            } else {
                const uint8_t*p=(const uint8_t*)W->data; uint32_t nb=0,nr=0; memcpy(&nb,p,4); memcpy(&nr,p+4,4);
                const uint8_t*scales=p+8,*packed=scales+nb,*res=packed+(size_t)nb*16u; uint32_t cur_blk=UINT32_MAX; float cur_sc=1.0f;
                for(int64_t k=0;k<K;++k){ const float xik=x[i*K+k];
                    for(int64_t j=c0;j<c1;++j){
                        int64_t idx=k*N+j; uint32_t blk=(uint32_t)(idx>>5); if(blk!=cur_blk){cur_blk=blk;cur_sc=ldexpf(1.0f,(int)scales[blk]-127);}
                        int off=(int)(idx&31); uint8_t byte=packed[(size_t)blk*16u+(size_t)(off>>1)];
                        c[i*N+j]+=xik*FP4_LUT[(off&1)?((byte>>4)&0x0F):(byte&0x0F)]*cur_sc;
                    }
                }
                if(W->reserved==1&&nr>0){
                    for(uint32_t ri=fp4_res_lb(res,nr,N,c0);ri<nr;++ri){
                        uint32_t idx; float diff; memcpy(&idx,res+(size_t)ri*8u,4); memcpy(&diff,res+(size_t)ri*8u+4,4);
                        int64_t rk=(int64_t)idx/N, rj=(int64_t)idx%N; if(rj>=c1)break; c[i*N+rj]+=x[i*K+rk]*diff;
                    }
                } else {
                    for(uint32_t ri=0;ri<nr;++ri){
                        uint32_t idx; float diff; memcpy(&idx,res+(size_t)ri*8u,4); memcpy(&diff,res+(size_t)ri*8u+4,4);
                        if((int64_t)idx<K*N){int64_t rk=(int64_t)idx/N,rj=(int64_t)idx%N;if(rj>=c0&&rj<c1)c[i*N+rj]+=x[i*K+rk]*diff;}
                    }
                }
            }
        }
    }
    if(R>64){free(sc);free(ord);}
    return AXON_OK;
}

static int bcast_strides(const AxonTensor*T,int64_t nd,const int64_t*shape,int64_t*st){
    for(int64_t i=0;i<nd;++i)st[i]=0; int64_t off=nd-T->ndim; if(off<0)return AXON_ERR_SHAPE;
    for(int64_t i=0;i<T->ndim;++i){ int64_t d=T->shape[i],s=shape[off+i]; if(d==s)st[off+i]=T->strides[i]; else if(d==1)st[off+i]=0; else return AXON_ERR_SHAPE; }
    return AXON_OK;
}
static int elementwise(const AxonTensor*A,const AxonTensor*B,AxonTensor*C,int mode){
    REQ(A&&B&&C&&A->data&&B->data&&C->data,AXON_ERR_ARG);
    REQ(A->dtype==AXON_DTYPE_F32&&B->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(C->ndim>=1&&C->ndim<=AXON_MAX_DIMS,AXON_ERR_SHAPE);
    const float*a=(const float*)A->data,*b=(const float*)B->data; float*c=(float*)C->data; const int64_t n=axon_tensor_numel(C);
    int same=(A->ndim==C->ndim&&B->ndim==C->ndim);
    if(same)for(int64_t i=0;i<C->ndim;++i)if(A->shape[i]!=C->shape[i]||B->shape[i]!=C->shape[i])same=0;
    if(same){
        #pragma omp parallel for schedule(static) if(n>=AXON_PAR_MIN)
        for(int64_t i=0;i<n;++i){float x=a[i],y=b[i]; c[i]=mode==0?x+y:(mode==1?x-y:x*y);} return AXON_OK;
    }
    int64_t sa[AXON_MAX_DIMS],sb[AXON_MAX_DIMS];
    REQ(bcast_strides(A,C->ndim,C->shape,sa)==AXON_OK,AXON_ERR_SHAPE); REQ(bcast_strides(B,C->ndim,C->shape,sb)==AXON_OK,AXON_ERR_SHAPE);
    for(int64_t lin=0;lin<n;++lin){ int64_t rem=lin,oa=0,ob=0;
        for(int64_t d=0;d<C->ndim;++d){ int64_t idx=rem/C->strides[d]; rem-=idx*C->strides[d]; oa+=idx*sa[d]; ob+=idx*sb[d]; }
        float x=a[oa],y=b[ob]; c[lin]=mode==0?x+y:(mode==1?x-y:x*y);
    }
    return AXON_OK;
}
int axon_add(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return elementwise(A,B,C,0);}
int axon_sub(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return elementwise(A,B,C,1);}
int axon_mul(const AxonTensor*A,const AxonTensor*B,AxonTensor*C){return elementwise(A,B,C,2);}
static int unary(const AxonTensor*A,AxonTensor*C,int mode){
    REQ(A&&C&&A->data&&C->data,AXON_ERR_ARG); REQ(A->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(axon_tensor_numel(A)==axon_tensor_numel(C),AXON_ERR_SHAPE);
    const float*a=(const float*)A->data; float*c=(float*)C->data; const int64_t n=axon_tensor_numel(A);
    #pragma omp parallel for schedule(static) if(n>=AXON_PAR_MIN)
    for(int64_t i=0;i<n;++i)c[i]=act_apply(a[i],mode); return AXON_OK;
}
int axon_relu(const AxonTensor*A,AxonTensor*C){return unary(A,C,1);}
int axon_gelu(const AxonTensor*A,AxonTensor*C){return unary(A,C,2);}
int axon_silu(const AxonTensor*A,AxonTensor*C){return unary(A,C,3);}
int axon_sigmoid(const AxonTensor*A,AxonTensor*C){return unary(A,C,4);}
int axon_softmax(const AxonTensor*A,AxonTensor*C){
    REQ(A&&C&&A->data&&C->data,AXON_ERR_ARG); REQ(A->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(A->ndim>=1&&A->ndim==C->ndim,AXON_ERR_SHAPE);
    for(int64_t i=0;i<A->ndim;++i)REQ(A->shape[i]==C->shape[i],AXON_ERR_SHAPE);
    const int64_t cols=A->shape[A->ndim-1]; REQ(cols>0,AXON_ERR_SHAPE);
    int64_t rows=1; for(int64_t i=0;i<A->ndim-1;++i)rows*=A->shape[i];
    const float*a=(const float*)A->data; float*c=(float*)C->data;
    #pragma omp parallel for schedule(static) if(rows*cols>=AXON_PAR_MIN)
    for(int64_t r=0;r<rows;++r){ const float*x=a+r*cols; float*y=c+r*cols; float mx=x[0];
        for(int64_t j=1;j<cols;++j)if(x[j]>mx)mx=x[j];
        float sum=0.f; for(int64_t j=0;j<cols;++j){float e=expf(x[j]-mx);y[j]=e;sum+=e;}
        float inv=sum>0.f?1.f/sum:0.f; for(int64_t j=0;j<cols;++j)y[j]*=inv;
    }
    return AXON_OK;
}
int axon_layer_norm(const AxonTensor*X,const AxonTensor*G,const AxonTensor*B,AxonTensor*C){
    REQ(X&&C&&X->data&&C->data,AXON_ERR_ARG); REQ(X->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(X->ndim>=1&&X->ndim==C->ndim,AXON_ERR_SHAPE);
    for(int64_t i=0;i<X->ndim;++i)REQ(X->shape[i]==C->shape[i],AXON_ERR_SHAPE);
    const int64_t cols=X->shape[X->ndim-1]; REQ(cols>0,AXON_ERR_SHAPE);
    int64_t rows=1; for(int64_t i=0;i<X->ndim-1;++i)rows*=X->shape[i];
    const float*g=NULL,*b=NULL;
    if(G){REQ(G->data&&G->ndim==1&&G->shape[0]==cols&&G->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE);g=(const float*)G->data;}
    if(B){REQ(B->data&&B->ndim==1&&B->shape[0]==cols&&B->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE);b=(const float*)B->data;}
    const float*x=(const float*)X->data; float*c=(float*)C->data;
    #pragma omp parallel for schedule(static) if(rows*cols>=AXON_PAR_MIN)
    for(int64_t r=0;r<rows;++r){ const float*xr=x+r*cols; float*yr=c+r*cols; float mean=0.f;
        for(int64_t j=0;j<cols;++j)mean+=xr[j]; mean/=(float)cols; float var=0.f;
        for(int64_t j=0;j<cols;++j){float d=xr[j]-mean;var+=d*d;}
        float inv=1.f/sqrtf(var/(float)cols+LN_EPS);
        for(int64_t j=0;j<cols;++j)yr[j]=((xr[j]-mean)*inv)*(g?g[j]:1.f)+(b?b[j]:0.f);
    }
    return AXON_OK;
}
int axon_transpose(const AxonTensor*A,AxonTensor*C){
    REQ(A&&C&&A->data&&C->data,AXON_ERR_ARG); REQ(A->ndim==2&&C->ndim==2,AXON_ERR_SHAPE);
    REQ(A->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    const int64_t M=A->shape[0],N=A->shape[1]; REQ(C->shape[0]==N&&C->shape[1]==M,AXON_ERR_SHAPE);
    const float*a=(const float*)A->data; float*c=(float*)C->data;
    #pragma omp parallel for collapse(2) schedule(static) if(M*N>=AXON_PAR_MIN)
    for(int64_t i=0;i<M;++i)for(int64_t j=0;j<N;++j)c[j*M+i]=a[i*N+j];
    return AXON_OK;
}
int axon_equilibrium(const AxonTensor*X,const AxonTensor*W,const AxonTensor*B,AxonTensor*C){
    REQ(X&&W&&C&&X->data&&W->data&&C->data,AXON_ERR_ARG);
    REQ(X->dtype==AXON_DTYPE_F32&&W->dtype==AXON_DTYPE_F32&&C->dtype==AXON_DTYPE_F32,AXON_ERR_DTYPE);
    REQ(X->ndim>=1&&X->ndim==C->ndim,AXON_ERR_SHAPE);
    for(int64_t i=0;i<X->ndim;++i)REQ(X->shape[i]==C->shape[i],AXON_ERR_SHAPE);
    const int64_t D=X->shape[X->ndim-1]; REQ(D>0&&W->ndim==2&&W->shape[0]==D&&W->shape[1]==D,AXON_ERR_SHAPE);
    const float*b=NULL; if(B){REQ(B->data&&B->ndim==1&&B->shape[0]==D&&B->dtype==AXON_DTYPE_F32,AXON_ERR_SHAPE);b=(const float*)B->data;}
    int64_t M=1; for(int64_t i=0;i<X->ndim-1;++i)M*=X->shape[i];
    const float*x=(const float*)X->data,*w=(const float*)W->data; float*s=(float*)C->data;
    memcpy(s,x,(size_t)(M*D)*sizeof(float));
    AxonTensor*tmp=axon_tensor_alloc(X->ndim,X->shape,AXON_DTYPE_F32); REQ(tmp&&tmp->data,AXON_ERR_NOMEM);
    float*sn=(float*)tmp->data;
    for(int step=0;step<16;++step){ float max_diff=0.f;
        for(int64_t i=0;i<M;++i)for(int64_t j=0;j<D;++j){
            float acc=x[i*D+j]+(b?b[j]:0.f); for(int64_t k=0;k<D;++k)acc+=s[i*D+k]*w[k*D+j];
            float v=0.7f*s[i*D+j]+0.3f*tanhf(acc); float d=fabsf(v-s[i*D+j]); if(d>max_diff)max_diff=d; sn[i*D+j]=v;
        }
        memcpy(s,sn,(size_t)(M*D)*sizeof(float)); if(max_diff<1e-4f)break;
    }
    axon_tensor_free(tmp); return AXON_OK;
}
int axon_bench_matmul(int n,double*out_ms,float*out_c){
    REQ(n>0&&out_ms&&out_c,AXON_ERR_ARG); int64_t shp[2]={n,n};
    AxonTensor*A=axon_tensor_alloc_persistent(2,shp,AXON_DTYPE_F32); AxonTensor*B=axon_tensor_alloc_persistent(2,shp,AXON_DTYPE_F32); AxonTensor*C=axon_tensor_alloc_persistent(2,shp,AXON_DTYPE_F32);
    if(!A||!B||!C){axon_tensor_free(A);axon_tensor_free(B);axon_tensor_free(C);return AXON_ERR_NOMEM;}
    float*a=(float*)A->data,*b=(float*)B->data;
    for(int64_t i=0;i<n;++i)for(int64_t j=0;j<n;++j){a[i*n+j]=(float)(i%10)*0.01f;b[i*n+j]=(float)(j%10)*0.01f;}
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0); int rc=axon_matmul(A,B,C); clock_gettime(CLOCK_MONOTONIC,&t1);
    *out_ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6;
    if(rc==AXON_OK)memcpy(out_c,C->data,(size_t)n*n*sizeof(float));
    axon_tensor_free(A);axon_tensor_free(B);axon_tensor_free(C); return rc;
}
#define LSH_BITS 6
#define LSH_BUCKETS (1<<LSH_BITS)
static inline uint32_t xs32(uint32_t*s){uint32_t x=*s;x^=x<<13;x^=x>>17;x^=x<<5;*s=x;return x;}
void axon_lsh_benchmark(int dim){
    if(dim<8){printf("dim must be >=8\n");return;}
    const int K=dim,N=dim,REPS=50;
    float*A=(float*)malloc((size_t)K*4),*B=(float*)malloc((size_t)K*N*4),*Cd=(float*)calloc((size_t)N,4),*Cl=(float*)calloc((size_t)N,4);
    int*cnt=(int*)calloc(LSH_BUCKETS,sizeof(int)),*cols=(int*)malloc((size_t)LSH_BUCKETS*N*sizeof(int));
    if(!A||!B||!Cd||!Cl||!cnt||!cols)goto done;
    for(int i=0;i<K;++i)A[i]=((i%7)-3)*0.1f;
    for(int i=0;i<K*N;++i)B[i]=((i%11)-5)*0.05f;
    for(int j=0;j<N;++j){ uint8_t h=0;
        for(int b=0;b<LSH_BITS;++b){ uint32_t rng=1337u+(uint32_t)b*7919u; float proj=0.f; for(int k=0;k<K;++k)proj+=B[(size_t)k*N+j]*((xs32(&rng)&1)?1.f:-1.f); if(proj>=0.f)h|=(uint8_t)(1u<<b); }
        cols[(size_t)h*N+cnt[h]++]=j;
    }
    struct timespec t0,t1,t2,t3; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int r=0;r<REPS;++r)for(int j=0;j<N;++j){float s=0.f;for(int k=0;k<K;++k)s+=A[k]*B[(size_t)k*N+j];Cd[j]=s;}
    clock_gettime(CLOCK_MONOTONIC,&t1);
    uint8_t hq=0; for(int b=0;b<LSH_BITS;++b){uint32_t rng=1337u+(uint32_t)b*7919u;float proj=0.f;for(int k=0;k<K;++k)proj+=A[k]*((xs32(&rng)&1)?1.f:-1.f);if(proj>=0.f)hq|=(uint8_t)(1u<<b);}
    clock_gettime(CLOCK_MONOTONIC,&t2);
    for(int r=0;r<REPS;++r){ memset(Cl,0,(size_t)N*4);
        for(int nb=0;nb<=LSH_BITS;++nb){ uint8_t hb=nb==0?hq:(uint8_t)(hq^(1u<<(nb-1))); int c=cnt[hb];
            for(int t=0;t<c;++t){int j=cols[(size_t)hb*N+t];float s=0.f;for(int k=0;k<K;++k)s+=A[k]*B[(size_t)k*N+j];Cl[j]=s;}
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&t3);
    int hit=0; double num=0.0,den=0.0;
    for(int j=0;j<N;++j){double d=Cd[j],l=Cl[j]; if(fabs(d)>1e-6&&fabs(l-d)<=1e-4*fabs(d))hit++; num+=(l-d)*(l-d);den+=d*d;}
    printf("=== AXON LSH (%dx%d) === dense: %.3fms | lsh: %.3fms | recall: %.1f%% | rel_L2: %.4f\n",K,N,
           (t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6,(t3.tv_sec-t2.tv_sec)*1e3+(t3.tv_nsec-t2.tv_nsec)/1e6,
           100.0*hit/N,den>0?sqrt(num/den):0.0);
done: free(A);free(B);free(Cd);free(Cl);free(cnt);free(cols);
}
'''

FILES["axon/runtime/axon_main.c"] = r'''#define _POSIX_C_SOURCE 200809L
#include "axon_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static void usage(const char*p){ printf("usage: %s [--input \"a,b,c\"] [--weights file.axn3] [--iters N] [--arena-mb N] [--max-print N] [--quiet]\n",p); }
static float* parse_input(const char*text,int64_t*count){
    size_t cap=64,n=0; float*v=(float*)malloc(cap*sizeof(float)); if(!v)return NULL;
    const char*p=text;
    while(*p){ char*end=NULL; float f=strtof(p,&end); if(end==p){free(v);return NULL;}
        if(n==cap){cap*=2;float*nv=(float*)realloc(v,cap*sizeof(float));if(!nv){free(v);return NULL;}v=nv;}
        v[n++]=f;p=end; while(*p==','||*p==' '||*p=='\t')++p;
    }
    *count=(int64_t)n; return v;
}
int main(int argc,char**argv){
    const char*input_text=NULL,*weights_path=NULL; int iters=1,quiet=0; size_t arena_mb=64; int64_t max_print=16;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")){usage(argv[0]);return 0;}
        else if(!strcmp(argv[i],"--input")&&i+1<argc)input_text=argv[++i];
        else if(!strcmp(argv[i],"--weights")&&i+1<argc)weights_path=argv[++i];
        else if(!strcmp(argv[i],"--iters")&&i+1<argc){iters=atoi(argv[++i]);if(iters<1)iters=1;}
        else if(!strcmp(argv[i],"--arena-mb")&&i+1<argc){long v=atol(argv[++i]);if(v>=0)arena_mb=(size_t)v;}
        else if(!strcmp(argv[i],"--max-print")&&i+1<argc){long v=atol(argv[++i]);if(v>0)max_print=(int64_t)v;}
        else if(!strcmp(argv[i],"--quiet"))quiet=1;
        else{fprintf(stderr,"unknown argument: %s\n",argv[i]);usage(argv[0]);return 2;}
    }
    if(weights_path){
        if(axon_load_weights_file(weights_path)!=AXON_OK){fprintf(stderr,"failed to mmap weights: %s\n",weights_path);return 2;}
    } else {
        const void*bp=NULL; int64_t bn=0; axon_weights_get(&bp,&bn);
        if(!bp){
            size_t len=strlen(argv[0])+6; char*ap=(char*)malloc(len);
            if(ap){ snprintf(ap,len,"%s.axn3",argv[0]);
                if(axon_load_weights_file(ap)!=AXON_OK){fprintf(stderr,"missing weights (tried %s)\n",ap);free(ap);return 2;}
                free(ap);
            }
        }
    }
    const int64_t ndim=axon_get_input_ndim(); int64_t shape[AXON_MAX_DIMS]={0};
    if(ndim<1||ndim>AXON_MAX_DIMS){fprintf(stderr,"bad input rank\n");axon_unload_weights_file();return 1;}
    axon_get_input_shape(shape); const int64_t need=axon_get_input_numel(); float*vals=NULL; int64_t got=0;
    if(input_text){ vals=parse_input(input_text,&got); if(!vals){fprintf(stderr,"bad --input\n");axon_unload_weights_file();return 2;}
        if(got!=need){fprintf(stderr,"input size mismatch: need %ld got %ld\n",(long)need,(long)got);free(vals);axon_unload_weights_file();return 2;}
    } else { got=need; vals=(float*)malloc((size_t)need*sizeof(float)); if(!vals){axon_unload_weights_file();return 1;}
        const float seed[4]={1.f,0.5f,-1.f,2.f}; for(int64_t i=0;i<need;++i)vals[i]=seed[i%4];
    }
    AxonTensor*inp=axon_tensor_alloc(ndim,shape,AXON_DTYPE_F32);
    if(!inp||axon_tensor_fill_f32(inp,vals,got)!=AXON_OK){fprintf(stderr,"input alloc failed\n");axon_tensor_free(inp);free(vals);axon_unload_weights_file();return 1;}
    free(vals); AxonTensor*inputs[1]={inp};
    if(arena_mb>0) axon_arena_init(arena_mb*1024u*1024u);
    AxonTensor*out=axon_forward(inputs);
    if(!out){fprintf(stderr,"inference failed\n");axon_shutdown();axon_tensor_free(inp);axon_arena_destroy();axon_unload_weights_file();return 1;}
    const int temp_out=(axon_output_is_temp()!=0); struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int it=0;it<iters;++it){
        if(out&&temp_out) axon_tensor_free(out);
        if(arena_mb>0) axon_arena_reset();
        out=axon_forward(inputs);
        if(!out){fprintf(stderr,"inference failed at %d\n",it);axon_shutdown();axon_tensor_free(inp);axon_arena_destroy();axon_unload_weights_file();return 1;}
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double us=((double)(t1.tv_sec-t0.tv_sec)*1e6+(double)(t1.tv_nsec-t0.tv_nsec)/1e3)/iters;
    int64_t flops=axon_get_total_flops(); double gflops=(us>0.0&&flops>0)?(double)flops/(us*1e3):0.0;
    if(!quiet){ printf("[axon] iters=%d arena=%zu/%zu bytes\n",iters,axon_arena_used(),axon_arena_capacity());
        axon_print_tensor_n(out,max_print); printf("[axon] avg %.2f us | est flops %ld | ~%.4f GFLOP/s (estimate)\n",us,(long)flops,gflops); }
    if(out&&temp_out) axon_tensor_free(out);
    axon_shutdown(); axon_tensor_free(inp); axon_arena_destroy(); axon_unload_weights_file();
    int64_t live=axon_live_allocs();
    if(!quiet)printf("[axon] live heap tensors at exit: %ld %s\n",(long)live,live==0?"(no leak)":"(LEAK!)");
    return live==0?0:1;
}
'''

FILES["axon/runtime/Makefile"] = r'''CC ?= cc
AR ?= ar
NATIVE ?= 0
SANITIZE ?= 0
OMP_CFLAGS ?=
OMP_LDFLAGS ?=
EXTRA ?=
ifeq ($(SANITIZE),1)
BASE = -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fPIC -std=c11 -Wall -Wextra -Wno-unknown-pragmas -fno-math-errno -pthread
LDFLAGS_SAN = -fsanitize=address,undefined
else
BASE = -O3 -fPIC -std=c11 -Wall -Wextra -Wno-unknown-pragmas -fno-math-errno -pthread
LDFLAGS_SAN =
endif
ifeq ($(NATIVE),1)
BASE += -march=native
endif
CFLAGS_LOCAL = $(BASE) $(OMP_CFLAGS) $(EXTRA)
.PHONY: all clean
all: libaxon_runtime.a libaxon_runtime.so axon_main.o
-include axon_runtime.d axon_main.d
axon_runtime.o: axon_runtime.c axon_runtime.h
@@$(CC) $(CFLAGS_LOCAL) -MMD -MP -c $< -o $@
axon_main.o: axon_main.c axon_runtime.h
@@$(CC) $(CFLAGS_LOCAL) -MMD -MP -c $< -o $@
libaxon_runtime.a: axon_runtime.o
@@$(AR) rcs $@ $^
libaxon_runtime.so: axon_runtime.o
@@$(CC) $(CFLAGS_LOCAL) -shared $^ $(LDFLAGS_SAN) $(OMP_LDFLAGS) -lm -o $@
clean:
@@rm -f *.o *.d *.a *.so .axon_tuning_stamp
'''

FILES["examples/transformer_block.axon"] = r'''fn transformer_block(x: f32[1, 4]) -> f32[1, 4] {
    let W_q: f32[4, 4] = [0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.2, 0.2, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.1] as f32[4, 4];
    let W_k: f32[4, 4] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] as f32[4, 4];
    let W_v: f32[4, 4] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5] as f32[4, 4];
    let q: f32[1, 4] = matmul(x, W_q); let k_t: f32[4, 4] = transpose(W_k);
    let scores: f32[1, 4] = matmul(q, k_t); let attn: f32[1, 4] = softmax(scores);
    let context: f32[1, 4] = matmul(attn, W_v);
    let W1: f32[4, 8] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0, 0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1] as f32[4, 8];
    let b1: f32[8] = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] as f32[8];
    let W2: f32[8, 4] = [0.1,0.2,0.3,0.4, 0.2,0.3,0.4,0.5, 0.3,0.4,0.5,0.6, 0.4,0.5,0.6,0.7, 0.5,0.6,0.7,0.8, 0.6,0.7,0.8,0.9, 0.7,0.8,0.9,1.0, 0.8,0.9,1.0,1.1] as f32[8, 4];
    let b2: f32[4] = [0.05,0.05,0.05,0.05] as f32[4];
    let h: f32[1, 8] = gelu(add(matmul(context, W1), b1));
    let ffn_out: f32[1, 4] = add(matmul(h, W2), b2);
    let gamma: f32[4] = [1.0,1.0,1.0,1.0] as f32[4]; let beta: f32[4] = [0.0,0.0,0.0,0.0] as f32[4];
    return layernorm(add(x, ffn_out), gamma, beta);
}
'''

FILES["examples/equilibrium.axon"] = r'''fn eq_net(x: f32[1, 4]) -> f32[1, 4] {
    let W: f32[4, 4] = [0.2,0.1,-0.1,0.0, 0.1,0.2,0.0,-0.1, -0.1,0.0,0.2,0.1, 0.0,-0.1,0.1,0.2] as f32[4, 4];
    let b: f32[4] = [5e-2,0.05,0.05,0.05] as f32[4];
    return equilibrium(x, W, b);
}
'''

FILES["examples/mlp.axon"] = r'''fn mlp(x: f32[1, 4]) -> f32[1, 2] {
    let W: f32[4, 4] = [1e-1,0.0,2e-1,0.0, 0.0,1e-1,0.0,2e-1, 2e-1,0.0,1e-1,0.0, 0.0,2e-1,0.0,1e-1] as f32[4, 4];
    let b: f32[4] = [0.1,-0.1,0.05,0.0] as f32[4];
    let h: f32[1, 4] = relu(matmul(x, W) + b);
    let scaled: f32[1, 4] = h * [2.0] + [-0.5];
    let O: f32[4, 2] = [0.5,0.0, 0.0,0.5, 0.25,0.25, 0.1,-0.1] as f32[4, 2];
    return softmax(matmul(scaled, O));
}
'''

FILES["tests/test_c_runtime.c"] = r'''#include "axon_runtime.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
int main(void) {
    axon_arena_init(1024 * 64);
    int64_t sx[2] = {2, 4}, sw[2] = {4, 8}, ss[2] = {2, 4}; (void)ss;
    int64_t ssig[2] = {4, 4};
    AxonTensor *X = axon_tensor_alloc(2, sx, AXON_DTYPE_F32);
    AxonTensor *W = axon_tensor_alloc(2, sw, AXON_DTYPE_F32);
    AxonTensor *S = axon_tensor_alloc(2, ssig, AXON_DTYPE_F32);
    AxonTensor *C = axon_tensor_alloc(2, (int64_t[2]){2, 8}, AXON_DTYPE_F32);
    assert(X && W && S && C);
    float xv[8] = {1,2,3,4, -1,0.5f,2,-0.5f};
    float wv[32]; for (int i = 0; i < 32; ++i) wv[i] = (float)(i % 7 - 3) * 0.1f;
    float sv[16] = {0}; for (int k = 0; k < 4; ++k) for (int r = 0; r < 4; ++r) sv[k*4+r] = (wv[k*8+2*r] + wv[k*8+2*r+1]) * 0.5f;
    assert(axon_tensor_fill_f32(X, xv, 8) == AXON_OK);
    assert(axon_tensor_fill_f32(W, wv, 32) == AXON_OK);
    assert(axon_tensor_fill_f32(S, sv, 16) == AXON_OK);
    assert(axon_neuro_matmul(X, W, S, C) == AXON_OK);
    assert(axon_silu(C, C) == AXON_OK);
    assert(axon_sigmoid(C, C) == AXON_OK);
    assert(axon_matmul(X, W, NULL) == AXON_ERR_ARG);
    assert(axon_load_weights_file("/nonexistent_axon_file.axn3") == AXON_ERR_ARG);
    axon_arena_reset();
    axon_arena_destroy();
    assert(axon_live_allocs() == 0);
    printf("PASS test_c_runtime (ASan/UBSan clean)\n");
    return 0;
}
'''

FILES["tests/test_axon.py"] = r'''import ctypes, os, shutil, struct, subprocess, tempfile, unittest, json
from axon.frontend import Lexer, Parser, Semantic
from axon.ir import IRBuilder, OpKind, WeightBlob
from axon.passes import Pipeline, FP4Quant, TernaryQuant, round_fp4
from axon.backend import LLVMBackend, Verilog, Baker
from axon.blob import BlobPacker, payload_bytes
from axon.reference import T, evaluate, dequant
from axon.runtime_paths import ensure_runtime
from axon.tools import Converter
from axon.errors import LexError, ParseError, SemanticError

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); EX=os.path.join(ROOT,"examples"); DELTA=2e-4

class _TensorStruct(ctypes.Structure):
    _fields_=[("data",ctypes.c_void_p),("ndim",ctypes.c_int64),("shape",ctypes.c_int64*4),("strides",ctypes.c_int64*4),
              ("dtype",ctypes.c_int32),("reserved",ctypes.c_int32),("scale",ctypes.c_float),("owns_data",ctypes.c_bool),("in_arena",ctypes.c_bool)]

def compile_src(src,fusion=True,quant=True,ternary=False,brain=False):
    ast=Parser(Lexer(src).tokenize()).parse(); Semantic().analyze(ast)
    return Pipeline(fusion=fusion,quant=quant,ternary=ternary,brain=brain).run(IRBuilder().build(ast))
def parse_output(text):
    vals=[]; inside=False
    for line in text.splitlines():
        s=line.strip()
        if s.endswith("= ["): inside=True; continue
        if inside and s=="]": break
        if inside:
            try: vals.append(float(s))
            except ValueError: pass
    return vals
def input_for(g):
    shape=g.get(g.inputs[0]).shape; n=1
    for d in shape: n*=d
    seed=[1.,2.,3.,4.]; return T(shape,[seed[i%4] for i in range(n)])
def close(a,b,delta=DELTA):
    if len(a)!=len(b): raise AssertionError(f"length {len(a)} != {len(b)}")
    for i,(x,y) in enumerate(zip(a,b)):
        if abs(x-y)>delta: raise AssertionError(f"index {i}: {x} != {y} (delta {delta})")

class Toolchain(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rt=ensure_runtime(); cls.tmp=tempfile.mkdtemp(prefix="axon_test_")
    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls.tmp,ignore_errors=True)
    def build(self,src,name,external_weights=False,**kw):
        g=compile_src(src,**kw); obj=os.path.join(self.tmp,name+".o"); exe=os.path.join(self.tmp,name+".bin")
        LLVMBackend().compile(g,obj); Baker().bake([obj],g.weights,exe,external_weights=external_weights); return g,exe
    def run_exe(self,exe,inp="1,2,3,4",iters=3,arena_mb=64,max_print=100000,extra_args=()):
        r=subprocess.run([exe,"--input",inp,"--iters",str(iters),"--arena-mb",str(arena_mb),"--max-print",str(max_print),*extra_args],capture_output=True,text=True,cwd=self.tmp)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr); self.assertIn("no leak",r.stdout); return parse_output(r.stdout)

class TestFrontend(unittest.TestCase):
    def test_rejects(self):
        cases=[
            ("fn f(x: f32[1,2]) -> f32[1,2] { return x; } fn g(y: f32[1,2]) -> f32[1,2] { return y; }",ParseError),
            ("fn f(x: f32[1,2]) -> f32[1,2] { let a=[1,2,3] as f32[1,2]; return x; }",SemanticError),
            ("fn f(x: f32[1,2]) -> f32[1,2] { return mul(x,x,x); }",SemanticError),
            ("fn f(x: f32[1,2]) -> f32[1,2] { return cast(x); }",SemanticError),
            ("fn f(x: f32[1,2,3,4,5]) -> f32[1,2] { return x; }",SemanticError),
            ("fn f(x: f32[1,0]) -> f32[1,2] { return x; }",SemanticError),
            ("fn f(x: f16[1,2]) -> f16[1,2] { return x; }",SemanticError),
            ("fn f(x: f32[1,2]) -> f32[1,3] { return x; }",SemanticError),
            ("fn f(x: f32[1,2]) -> f32[1,2] { let x=x; return x; }",SemanticError),
            ("fn f(x: f32[1,2]) -> f32[1,2] { let y=nope(x); return y; }",SemanticError),
            ("fn f(x: f32[1,2]) -> f32[1,2] { return x @ x; }",LexError),
        ]
        for src,exc in cases:
            with self.subTest(src=src[:40]):
                with self.assertRaises(exc): compile_src(src)
    def test_precedence(self):
        g=compile_src("fn f(x: f32[1,2]) -> f32[1,2] { return x * [2.0] + [1e-1,-2.5E+0]; }")
        self.assertEqual(g.get(g.outputs[0]).op,OpKind.ADD)
    def test_broadcast(self):
        g=compile_src("fn f(x: f32[4,1]) -> f32[4,3] { return x + [1.0,2.0,3.0]; }")
        self.assertEqual(g.get(g.outputs[0]).shape,(4,3))
    def test_dce(self):
        g=compile_src("fn f(x: f32[1,2]) -> f32[1,2] { let dead: f32[2,2]=[1,2,3,4] as f32[2,2]; let t=transpose(dead); return x; }")
        self.assertEqual(len(g.weights),0)

class TestQuant(unittest.TestCase):
    def test_fp4_outlier_sorted(self):
        data=[0.1*i for i in range(64)]; data[5]=150.0; data[33]=-90.0
        b=WeightBlob("w",data,[8,8],"f32"); FP4Quant()._q(b); rec=dequant(b)
        self.assertAlmostEqual(rec[5],150.0,delta=1e-3); self.assertAlmostEqual(rec[33],-90.0,delta=1e-3)
        cols=[idx%8 for idx,_ in b.residuals]; self.assertEqual(cols,sorted(cols))
    def test_rounding(self): self.assertEqual(round_fp4(0.25),0.5); self.assertEqual(round_fp4(-0.25),-0.5)
    def test_shared_weight_not_ternary(self):
        src="fn f(x: f32[1,2]) -> f32[1,2] { let W: f32[2,2]=[1,2,3,4] as f32[2,2]; let a=matmul(x,W); let t=transpose(W); return a + matmul(x,t); }"
        g=compile_src(src,ternary=True); self.assertEqual(g.alias_weights()["W"].quant,"none")
    def test_fp4_zerocopy_compressed(self):
        if os.environ.get("AXON_SANITIZE","0")=="1": return
        rt=ensure_runtime(); lib=ctypes.CDLL(str(rt.shared_lib))
        lib.axon_set_weights_blob.argtypes=[ctypes.c_void_p,ctypes.c_int64]
        lib.axon_weight_tensor.argtypes=[ctypes.c_char_p]; lib.axon_weight_tensor.restype=ctypes.POINTER(_TensorStruct)
        lib.axon_tensor_free.argtypes=[ctypes.c_void_p]
        b=WeightBlob("w",[0.1*i for i in range(32)],[8,4],"f32"); FP4Quant()._q(b)
        packed=BlobPacker.pack({"w":b}); buf=ctypes.create_string_buffer(packed,len(packed))
        lib.axon_set_weights_blob(ctypes.cast(buf,ctypes.c_void_p),len(packed))
        tp=lib.axon_weight_tensor(b"w"); self.assertTrue(bool(tp))
        self.assertEqual(tp.contents.dtype,2); self.assertFalse(tp.contents.owns_data); self.assertEqual(tp.contents.reserved,1)
        lib.axon_tensor_free(tp)
    def test_payload_accounting(self):
        b=WeightBlob("w",[0.1*i for i in range(16)],[4,4],"f32"); TernaryQuant()._q(b)
        self.assertEqual(payload_bytes(b),8+len(b.packed))

class TestBlob(unittest.TestCase):
    def test_header(self):
        b=WeightBlob("w",[float(i) for i in range(32)],[8,4],"f32"); packed=BlobPacker.pack({"w":b})
        magic,ver,count,_,total=struct.unpack_from("<4sIIIQ",packed,0)
        self.assertEqual(magic,b"AXN3"); self.assertEqual(ver,3); self.assertEqual(count,1); self.assertEqual(total,len(packed))
        _,noff,doff,nbytes,dtype,ndim=struct.unpack_from("<6Q",packed,24)
        self.assertEqual(dtype,0); self.assertEqual(ndim,2); self.assertEqual(doff%16,0)
    def test_corrupt_rejected(self):
        if os.environ.get("AXON_SANITIZE","0")=="1": return
        rt=ensure_runtime(); lib=ctypes.CDLL(str(rt.shared_lib))
        lib.axon_set_weights_blob.argtypes=[ctypes.c_void_p,ctypes.c_int64]; lib.axon_weight_tensor.argtypes=[ctypes.c_char_p]; lib.axon_weight_tensor.restype=ctypes.c_void_p
        b=WeightBlob("w",[1.,2.,3.,4.],[2,2],"f32"); packed=bytearray(BlobPacker.pack({"w":b})); packed[0]=0
        buf=ctypes.create_string_buffer(bytes(packed),len(packed)); lib.axon_set_weights_blob(ctypes.cast(buf,ctypes.c_void_p),len(packed))
        self.assertIsNone(lib.axon_weight_tensor(b"w"))
    def test_name_lookup_roundtrip(self):
        if os.environ.get("AXON_SANITIZE","0")=="1": return
        rt=ensure_runtime(); lib=ctypes.CDLL(str(rt.shared_lib))
        lib.axon_set_weights_blob.argtypes=[ctypes.c_void_p,ctypes.c_int64]; lib.axon_weight_tensor.argtypes=[ctypes.c_char_p]; lib.axon_weight_tensor.restype=ctypes.c_void_p
        lib.axon_tensor_free.argtypes=[ctypes.c_void_p]
        b=WeightBlob("w",[1.,2.,3.,4.],[2,2],"f32"); packed=BlobPacker.pack({"w":b})
        buf=ctypes.create_string_buffer(packed,len(packed)); lib.axon_set_weights_blob(ctypes.cast(buf,ctypes.c_void_p),len(packed))
        tp=lib.axon_weight_tensor(b"w"); self.assertIsNotNone(tp); lib.axon_tensor_free(ctypes.c_void_p(tp))

class TestEndToEnd(Toolchain):
    def test_f32_reference(self):
        src=open(os.path.join(EX,"transformer_block.axon")).read(); g,exe=self.build(src,"f32",quant=False)
        close(self.run_exe(exe),evaluate(g,[input_for(g)]).data)
    def test_fusion_identical(self):
        src=open(os.path.join(EX,"transformer_block.axon")).read()
        _,fused=self.build(src,"fused",quant=False); _,plain=self.build(src,"plain",fusion=False,quant=False)
        close(self.run_exe(fused),self.run_exe(plain),1e-5)
    def test_fp4_reference(self):
        src=open(os.path.join(EX,"transformer_block.axon")).read(); g,exe=self.build(src,"fp4",quant=True)
        close(self.run_exe(exe),evaluate(g,[input_for(g)]).data)
    def test_ternary_reference(self):
        src=open(os.path.join(EX,"transformer_block.axon")).read(); g,exe=self.build(src,"tern",ternary=True,quant=False)
        close(self.run_exe(exe),evaluate(g,[input_for(g)]).data)
    def test_brain_mmap_all_dtypes(self):
        src=open(os.path.join(EX,"transformer_block.axon")).read()
        for tag,kw in (("f32",{"quant":False}),("fp4",{"quant":True}),("tern",{"ternary":True,"quant":False})):
            with self.subTest(dtype=tag):
                g,exe=self.build(src,f"brain_{tag}",external_weights=True,brain=True,**kw)
                self.assertTrue(os.path.exists(exe+".axn3"))
                self.assertTrue(any(n.op==OpKind.NEURO_MATMUL for n in g.nodes))
                out=self.run_exe(exe,extra_args=("--weights",exe+".axn3"))
                close(out,evaluate(g,[input_for(g)]).data)
                out_auto=self.run_exe(exe)  # auto-discovers <exe>.axn3
                close(out_auto,out,1e-6)
    def test_silu_sigmoid_fusion(self):
        src="fn f(x: f32[1, 4]) -> f32[1, 4] {\n let W: f32[4, 4]=[0.2,0.1,-0.1,0.3,-0.2,0.4,0.1,-0.1,0.1,-0.3,0.2,0.2,0.3,0.0,-0.2,0.1] as f32[4, 4];\n let b: f32[4]=[0.1,-0.2,0.3,-0.1] as f32[4];\n return sigmoid(silu(add(matmul(x, W), b)));\n}"
        g,exe=self.build(src,"silu",quant=False); self.assertTrue(any(n.op==OpKind.FUSED_MATMUL_BIAS_SILU for n in g.nodes))
        close(self.run_exe(exe),evaluate(g,[input_for(g)]).data)
    def test_odd_prime_matmul_all_dtypes(self):
        M,K,N=19,37,29
        w_vals=[((i*7+j*13)%17-8)*0.05 for i in range(K) for j in range(N)]; w_vals[42]=25.0
        b_vals=[((j%5)-2)*0.1 for j in range(N)]
        src=f"fn prime_mm(x: f32[{M}, {K}]) -> f32[{M}, {N}] {{\n let W: f32[{K}, {N}]=[{', '.join(repr(v) for v in w_vals)}] as f32[{K}, {N}];\n let b: f32[{N}]=[{', '.join(repr(v) for v in b_vals)}] as f32[{N}];\n return silu(add(matmul(x, W), b));\n}}"
        inp_str=",".join(str([1.,2.,3.,4.][i%4]) for i in range(M*K))
        for tag,kw in (("f32",{"quant":False}),("fp4",{"quant":True}),("tern",{"ternary":True,"quant":False})):
            with self.subTest(dtype=tag):
                g,exe=self.build(src,f"prime_{tag}",**kw)
                out=self.run_exe(exe,inp=inp_str,iters=2,arena_mb=0,max_print=100000)
                ref=evaluate(g,[input_for(g)]); self.assertEqual(len(out),M*N); close(out,ref.data,5e-4)
    def test_equilibrium_reference(self):
        src=open(os.path.join(EX,"equilibrium.axon")).read(); g,exe=self.build(src,"eq",quant=False)
        close(self.run_exe(exe),evaluate(g,[input_for(g)]).data)
    def test_mlp_operators(self):
        src=open(os.path.join(EX,"mlp.axon")).read(); g,exe=self.build(src,"mlp",quant=False)
        out=self.run_exe(exe); close(out,evaluate(g,[input_for(g)]).data); self.assertAlmostEqual(sum(out),1.0,delta=1e-5)
    def test_no_arena_no_leak(self):
        src=open(os.path.join(EX,"transformer_block.axon")).read(); _,exe=self.build(src,"noarena",quant=False)
        self.run_exe(exe,iters=5,arena_mb=0)
    def test_input_size_mismatch(self):
        src=open(os.path.join(EX,"transformer_block.axon")).read(); _,exe=self.build(src,"mismatch",quant=False)
        r=subprocess.run([exe,"--input","1,2,3"],capture_output=True,text=True,cwd=self.tmp); self.assertNotEqual(r.returncode,0); self.assertIn("mismatch",r.stderr)
    def test_verilog(self):
        g=compile_src(open(os.path.join(EX,"transformer_block.axon")).read()); path=os.path.join(self.tmp,"chip.v"); Verilog.compile(g,path)
        v=open(path).read(); self.assertIn("module axon_mac_core",v); self.assertIn("ROUND_BIAS",v)
        tb=open(os.path.join(self.tmp,"chip_tb.v")).read(); self.assertIn("$readmemh",tb)
        lines=[l for l in open(os.path.join(self.tmp,"chip_weights.hex")).read().splitlines() if l.strip()]; self.assertEqual(len(lines),4)
    def test_safetensors_converter(self):
        w=struct.pack("<8f",*[0.1]*8); hdr=json.dumps({"layers.0.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,len(w)]}}).encode()
        path=os.path.join(self.tmp,"m.safetensors")
        with open(path,"wb") as f: f.write(struct.pack("<Q",len(hdr))); f.write(hdr); f.write(w)
        model=Converter.from_safetensors(path); src=Converter.generate(model); self.assertIn("matmul",src); compile_src(src)

if __name__=="__main__": unittest.main()
'''

LEGACY_PKG_DIRS = ["axon/frontend", "axon/ir", "axon/backend", "axon/tools", "axon/baker", "axon/optimizer", "axon/utils"]
LEGACY_FILES = ["tests/test_axon_suite.py"]

def _purge_legacy():
    for d in LEGACY_PKG_DIRS:
        if os.path.isdir(d): shutil.rmtree(d, ignore_errors=True)
    if os.path.isfile(os.path.join("runtime", "axon_tensor.c")): shutil.rmtree("runtime", ignore_errors=True)
    for f in LEGACY_FILES:
        if os.path.isfile(f): os.remove(f)
    for root, dirs, _ in os.walk("."):
        for d in list(dirs):
            if d == "__pycache__": shutil.rmtree(os.path.join(root, d), ignore_errors=True); dirs.remove(d)

def main():
    print("[1/5] Cleaning legacy directories ...")
    _purge_legacy()
    print("[2/5] Unpacking Axon v7.0 ...")
    for path, content in FILES.items():
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        if path.endswith("Makefile"): content = content.replace("@@", "\t")
        with open(path, "w", encoding="utf-8") as f: f.write(content)
    print("[3/5] Building runtime ...")
    from axon.runtime_paths import ensure_runtime
    ensure_runtime(verbose=True)
    print("[4/5] Running unit tests ...")
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"])
    if r.returncode != 0: sys.exit(1)
    print("[5/5] Building v7.0 examples (standalone, brain+mmap, ternary, RTL) ...")
    for cmd in (
        [sys.executable, "-m", "axon.cli", "build", "examples/transformer_block.axon", "-o", "transformer.bin"],
        [sys.executable, "-m", "axon.cli", "build", "examples/transformer_block.axon", "-o", "transformer_brain.bin", "--brain", "--external-weights"],
        [sys.executable, "-m", "axon.cli", "build", "examples/transformer_block.axon", "-o", "transformer_ternary.bin", "--ternary"],
        [sys.executable, "-m", "axon.cli", "build", "examples/mlp.axon", "-o", "mlp.bin", "--no-quant"],
        [sys.executable, "-m", "axon.cli", "build", "examples/transformer_block.axon", "-o", "axon_chip.v", "--emit-silicon"],
    ):
        subprocess.run(cmd, check=True)
    subprocess.run(["./transformer_brain.bin", "--input", "1.5,-0.5,2.0,0.8", "--iters", "100"], check=True)
    print("\n[COMPLETED] Axon v7.0 ready. Run: python3 -m axon.cli selfcheck && make sanitize")

if __name__ == "__main__":
    main()
