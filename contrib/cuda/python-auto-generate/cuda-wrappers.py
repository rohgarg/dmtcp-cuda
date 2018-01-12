#!/usr/bin/python

import sys
import os
import warnings

if len(sys.argv) != 2:
  print "***  Usage: " + sys.argv[0] + " <CUDA_WRAPPER_FILE.template>\n"
  sys.exit(1)

wrapper_file = open(sys.argv[1])
wrapper_declarations = wrapper_file.read()
wrapper_file.close()

# ===================================================================
# FORM LEXICAL TOKENS
# INPUT: wrapper_declarations
# OUTPUT: lexified

# We include ':' as a word char for sake of keywords: :IN, :OUT, etc.
word_ords = (range(ord('a'), ord('z')+1) + range(ord('A'),ord('Z')+1) +
             [ord('_')] + [ord(':')] + range(ord('0'),ord('9')+1))
word_chars = ''.join([chr(x) for x in word_ords])

ops = ["==", "+=", "*=", "-=", "/=", "%=", "!=", "&&", "||"]

def get_comment(input):
  if input[0] == "#":
    return input[:input.find('\n')] + '\n'
  elif len(input) >= 2 and input[:2] == "//":
    return input[:input.find('\n')] + '\n'
  elif len(input) >= 2 and input[:2] == "/*":
    return input[:input.find('*/')] + '*/'
  else:
    return ""
def is_comment(input):
  return input[0] == "#" or  len(input) >= 2 and input[:2] in ["//", "/+"]

def get_word(input):
  if len(input) >= 2 and input[:2] in ops:
    return input[:2]
  elif len(input) >= 2 and input[:2] == "//":
    return input[:input.find('\n')]
  elif input[0] in '"\'':
    quote = input[0]
    return quote + input[1:].split(quote)[0] + quote
  elif input[0] in word_chars:
    return input[0 : len(input)-len(input.lstrip(word_chars))]
  else:
    return input[0:1]  # Must return a string

myinput = wrapper_declarations
lexified = []
while(myinput):
  myinput = myinput.strip()
  comment = get_comment(myinput)
  if comment:
    word = comment
  else:
    word = get_word(myinput)
  myinput = myinput[len(word):]
  lexified.append(word)

# ===================================================================
# PARSE DECLARATION
# INPUT:  lexified
# OUTPUT: ast_wrappers (ast := Abstract Syntax Tree)

verbatim_wrappers = ""
verbatim_proxies = ""

def extract_balanced_expression(input):
  index = 1
  assert input[index] == '('
  expr = ""
  parens = 0
  while index <= 1 or parens > 0:
    if input[index] == '(':
      parens += 1
    elif input[index] == ')':
      parens -= 1
    expr += input[index] + ' '
    index += 1
  expr = expr.strip()[1:-1] # Strip expr of first '(' and final ')'
  expr = expr.replace(" {", "\n{").replace("; ", ";\n") + '\n'
  return (expr, input[index:])

myinput = lexified
myinput = [ term for term in myinput if not is_comment(term) ]
def parse(input):
  (p,r) = parse_aux(input, [])
  return p
def parse_aux(input, rest):
  global verbatim_wrappers, verbatim_proxies
  parsed = []
  delims = [',', '(', ')', ';']
  while(input):
    while input[0] in ["CUDA_VERBATIM_WRAPPER", "CUDA_VERBATIM_PROXY"]:
      verbatim_type = input[0]
      (expr, input) = extract_balanced_expression(input)
      if verbatim_type == "CUDA_VERBATIM_WRAPPER":
        verbatim_wrappers+= "// Verbatim from " + sys.argv[1] + "\n" + expr+"\n"
      elif verbatim_type == "CUDA_VERBATIM_PROXY":
        verbatim_proxies += "// Verbatim from " + sys.argv[1] + "\n" + expr+"\n"
    all_delims = [x for x in input if x in delims]
    next_delim = all_delims and all_delims[0]
    if not next_delim:
      parsed.append(input)
      input = []
    elif input[0] in [',', ';']:
      input = input[1:]
    elif input[0] == ')':
      # Somebody passed down "...)" from "(...)" at upper level.
      # Truncate rest of input, and force return to upper level.
      parsed.append(')')
      rest = input[1:] + rest
      input = []
    elif input[0] != next_delim:
      parsed.append(input[:input.index(next_delim)])
      input = input[input.index(next_delim):]
    elif input[0] == '(':
      if ')' not in input:
        print("Bad parse (No matching closing parenthesis found.):" + input)
        sys.exit(1)
      (p,r) = parse_aux(input[1:], rest)
      parsed.append(['('] + p)
      input = r
      rest = []
    else:
      print("Bad parse:" + input)
      sys.exit(1)
  return (parsed, rest)

ast_wrappers = parse(myinput)
#For debugging:
# print ast_wrappers

# ===================================================================
# CONVERT AST INTO ANNOTATED AST
# INPUT:  ast_wrappers (ast := Abstract Syntax Tree)
# OUTPUT: ast_annotated_wrappers

def generate_annotated_wrapper(isLogging, flushDirtyPages, ast):  # abstract syntax tree
  fnc = { "type" : ' '.join(ast[0][:-1]), "name" : ast[0][-1],
          "isLogging" : isLogging,
          "flushDirtyPages": flushDirtyPages}
  args = []
  raw_args = ast[1:][0][1:-1]  # Omit '(' and ')'
  # For 'int foo(void)', 'void' was parsed as a separate arg; remove it now.
  raw_args = [arg for arg in raw_args if arg != ["void"]]
  while raw_args:
    arg = raw_args[0]
    # If an optional C++ argument, "type param = 0", then remove "= ...".
    if '=' in arg:
      (arg, optional) = (arg[:arg.index('=')],
                         ' ' + ' '.join(arg[arg.index('='):]))
    else:
      optional = ""
    # First capture and delete any tags
    if isinstance(arg, list) and arg[0].startswith(':') and len(arg) == 1 and \
       isinstance(raw_args[1], list) and raw_args[1][0] == "(":
      # example: :IN_DEEP_COPY(len) ...
      #       => [[":IN_DEEP_COPY"], ["(", ["len"], ")"] ...]
      tag = [ arg[0][1:], raw_args[1][1][0] ]
      del raw_args[0:2]
      arg = raw_args[0]
    elif isinstance(arg[0], str) and arg[0].startswith(':'):
      # example: [:IN ...]
      tag = [arg[0][1:]] # Strip initial ':'
      del arg[0]
    else:
      tag = ["IN"]  # Default value
    args.append( { "tag" : tag , "type" : ' '.join(arg[:-1]),
                   "optional" : optional , "name" : arg[-1] } )
    del raw_args[0]
  return { "FNC" : fnc , "ARGS" : args }

ast_annotated_wrappers = []
while ast_wrappers:
  # ast_wrappers[0] : ['CUDA_WRAPPER' '(' FNC ARGS ')']
  if ast_wrappers[0][0] in ["CUDA_WRAPPER", "CUDA_WRAPPER_WITH_LOGGING", "CUDA_WRAPPER_WITH_UVM_SYNC"]:
    isLogging = ast_wrappers[0][0] == "CUDA_WRAPPER_WITH_LOGGING"
    flushDirtyPages = ast_wrappers[0][0] == "CUDA_WRAPPER_WITH_UVM_SYNC"
    del ast_wrappers[0]
    ast_annotated_wrappers.append( generate_annotated_wrapper( isLogging, flushDirtyPages,
                                                        ast_wrappers[0][1:3]) )
    del ast_wrappers[0]
  else:
    warnings.warn("Expression is not equal to CUDA_WRAPPER")
    del ast_wrappers[0:2]

#For debugging:
# print ast_annotated_wrappers

# ===================================================================
# EMIT GENERATED CODE
# INPUT:  ast_annotated_wrappers
# OUTPUT FILES: *-cudawrappers.icpp *-cudawrappers.icu *-cudawrappers.ih

# ====
# Utilities for CudaMemcpy, cudaMallocHost, etc.

#  FIXME:  Host memory can be allocated using
#          either cudaMallocHost() or cudaHostAlloc() or malloc().  But note
#          that cudaMallocHost(), etc. can be pinnned memory in proxy process.
def cudaMemcpyExtraCode(args, isLogging):
  (application_before, application_after, proxy_before, proxy_after) = 4*[""]
  args_dict = {}
  for key in ["DEST", "SRC", "IN_BUF", "SIZE",
              "DEST_PITCH", "SRC_PITCH", "HEIGHT", "DIRECTION"]:
    args_dict[key] = None  # Default for standard keys is: None
  for arg in args:
    args_dict[arg["tag"][0]] = arg["name"]

  if args_dict["IN_BUF"]:  # if "IN_BUF" and "SIZE" exist
    application_before += (
"""  JASSERT(write(skt_master, %s, %s) == %s) (JASSERT_ERRNO);
""" % (args_dict["IN_BUF"], args_dict["SIZE"], args_dict["SIZE"]))
    proxy_before += (
"""  // Allocate memory for IN_BUF arguments and receive data
  %s = malloc(%s);
  assert(read(skt_accept, %s, %s) == %s);
""" % (args_dict["IN_BUF"], args_dict["SIZE"],
       args_dict["IN_BUF"], args_dict["SIZE"], args_dict["SIZE"]))
    proxy_after += (
"""
  // Free the buffer for IN_BUF arguments
  free(%s);
""" % args_dict["IN_BUF"])

  #====
  # Generate non-trivial extra code only if "DIRECTION" tag was specified.
  if not args_dict["DIRECTION"]:
    return (application_before, application_after, proxy_before, proxy_after)

  assert args_dict["DIRECTION"] != "_direction"  # Check no name clash
  proxy_before += (
"""  enum cudaMemcpyKind _direction;
""")

  assert "_size" not in args_dict.values()  # Avoid name clash
  application_before += (
"""
  int _size = -1;
""")

  if args_dict["SIZE"]:
    assert not args_dict["HEIGHT"]
  else:
    assert len([True for key in ["DEST_PITCH", "SRC_PITCH", "HEIGHT"]
                     if key in args_dict.keys()]) == 3

  if args_dict["SIZE"]:
    size_appl_send = size_proxy_recv = (
"""_size = %s;""") % args_dict["SIZE"]
  else:
    size_appl_send = size_proxy_recv = (
"""_size = %s * %s;""") % (args_dict["SRC_PITCH"], args_dict["HEIGHT"])

  if args_dict["SIZE"]:
    size_proxy_send = size_appl_recv = (
"""_size = %s;""") % args_dict["SIZE"]
  else:
    size_proxy_send = size_appl_recv = (
"""_size = %s * %s;""") % (args_dict["DEST_PITCH"], args_dict["HEIGHT"])

  if args_dict["DIRECTION"]:  # if "DIRECTION" and "SRC" exist
    application_before += (
"""  %s
  if (_direction == cudaMemcpyHostToDevice ||
      _direction == cudaMemcpyHostToHost) {
    // Send source buffer to proxy process
    // NOTE: As an optimization, HostToHost could be done locally.
    // NOTE:  This assumes no pinnned memory.
    JASSERT(write(skt_master, %s, _size) == _size) (JASSERT_ERRNO);
  }
""" % (size_appl_send, args_dict["SRC"]))
  # NOTE:  We should not be logging these large buggers.
  #   We should only log CUDA fnc'only with prim. args; e.g., not cudaMemcpy()
  application_before = application_before[:-1] # Remove last brace of '{...}'
  if (isLogging):
    # Add '{' in comment, so editors will see balanced braces.
    cudawrappers.write(
"""log_append(%s, _size);
  }
""" % args_dict["SRC"])

  if args_dict["DIRECTION"]:  # if "DIRECTION" and "SRC" exist
    direction_declared = True
    proxy_before += (
"""  int _size = -1;
  %s
  _direction = %s;
  if (_direction == cudaMemcpyHostToDevice ||
      _direction == cudaMemcpyHostToHost) {
    // Receive source buffer from application process
    %s = malloc(_size);
    assert(read(skt_accept, %s, _size) == _size);
    // Get ready for receiving memory from device when making CUDA call
  }
  else if (_direction == cudaMemcpyDeviceToHost) {
    %s
    // NEEDED FOR DeviceToHost; SHOULD REUSE OLD malloc() for HostToHost
    // NOTE:  This assumes no pinnned memory.
    %s = malloc(_size);
  }
""" % (size_proxy_recv, args_dict["DIRECTION"], args_dict["SRC"],
       args_dict["SRC"], size_proxy_send, args_dict["DEST"]))

  if args_dict["DIRECTION"]:  # if "DIRECTION" and "DEST" exist
    if direction_declared:
      proxy_after += (
"""  _direction = %s;
""" % args_dict["DIRECTION"])
    else:
      proxy_after += (
"""  _direction = %s;
""" % args_dict["DIRECTION"])

    proxy_after += (
"""  if (_direction == cudaMemcpyDeviceToHost ||
      _direction == cudaMemcpyHostToHost) {
    // Send  dest buffer to application process
    // NOTE:  This assumes no pinnned memory.
    assert(write(skt_accept, %s, _size) == _size);
    free(%s);
  }
  else if (_direction == cudaMemcpyHostToDevice) {
    free(%s);
  }
""" % (args_dict["DEST"], args_dict["DEST"],
       args_dict["SRC"]))

  if args_dict["DIRECTION"]:  # if "DIRECTION" and "DEST" exist
    application_after += (
"""  %s
  if (_direction == cudaMemcpyDeviceToHost ||
      _direction == cudaMemcpyHostToHost) {
    // Receive dest buffer from proxy process
    // NOTE:  This assumes no pinnned memory.
    JASSERT(read(skt_master, %s, _size) == _size) (JASSERT_ERRNO);
  }
""" % (size_appl_recv, args_dict["DEST"]))

  return (application_before, application_after, proxy_before, proxy_after)
  # End of 'cudaMemcpyExtraCode(args, isLogging)'

# Not currently used
def MEMCPY(dest, source, size=None, buf_offset=None):
  result = "memcpy(" + dest + ", " + source
  if not size:
    variable = [param for param in [dest, source] if '+' not in param]
    assert len(variable) == 1
    size = "sizeof *" + variable[0]
#include <assert.h>
  result += ", " + size
  result += ")\m"
  if buf_offset:
    result += buf_offset + "+=" + size
#include <cuda_profiler_api.h>
  return result

# ====
# Emit code into files

if sys.argv[1] == "main.template":
  basefilename = ""
else:
  # For "foo.template", basefilename is "foo-"
  basefilename = sys.argv[1].rsplit('.',1)[0] + "-"
cuda_include = open(basefilename + "cuda_plugin.h", 'w+')
cuda_include2 = open(cuda_include.name + "-part2", 'w+')
cudawrappers = open(basefilename + "cudawrappers.icpp", 'w+')
cudaproxy = open(basefilename + "cudaproxy.icu", 'w+')
cudaproxy2 = open(cudaproxy.name + "-part2", 'w+')

cuda_include_head = (
"// Generated by " + sys.argv[0] + " and " + sys.argv[1] + """
#ifndef _CUDA_PLUGIN_H_
#define _CUDA_PLUGIN_H_

#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_plugin.h"
#include "jassert.h"

#define PYTHON_AUTO_GENERATE 1

// for pinned memory
#define PINNED_MEM_MAX_ALLOC 100
// extern typedef struct pseudoPinnedMem pseudoPinnedMem_t;
void pseudoPinnedMem_append(void *ptr);
bool is_pseudoPinnedMem(void *ptr);
void pseudoPinnedMem_remove(void *ptr);

extern int skt_master;
extern int skt_accept;
extern int initialized;

void proxy_initialize();

enum cuda_op {
""")
cuda_include.write(cuda_include_head)
cuda_include_tail = (
"""  OP_LAST_FNC
};""")

cudawrappers_head = (
"// Generated by " + sys.argv[0] + " and " + sys.argv[1] + """
#include """ + '"' + cuda_include.name + '"' + """

EXTERNC enum cudaMemcpyKind
cudaMemcpyGetDirection(const void *destPtr, const void *srcPtr,
                       enum cudaMemcpyKind *direction);

""")
cudawrappers.write(cudawrappers_head)
cudawrappers_tail = ""

cudaproxy_head = (
"// Generated by " + sys.argv[0] + " and " + sys.argv[1] + """
#include """ + '"' + cuda_include.name + '"' + """

extern void do_work();

int main() {
  // FIXME: ADD: do_init()
  do_work();
}

void do_work() {
  while(1) {
    enum cuda_op op;

    assert(read(skt_accept, &op, sizeof op) == sizeof op);
    switch (op) {
""")
cudaproxy.write(cudaproxy_head)
cudaproxy_tail = (
"""    }
  }
}
""")

# ====
# Write a wrapper fnc., a proxy stub fnc, and .h entries for each template fnc

def write_all_files(ast):  # abstract syntax tree
  fnc = ast["FNC"]
  args = ast["ARGS"]
  write_cuda_bodies(fnc, args) # write into cudawrapper and cudaproxy

def write_cuda_bodies(fnc, args):
  # For cudawrappers
  # NOTE:  We purposely ignore arg["optional"], and we make the optional
  #        argument _required_.  This works because the CUDA C++ library
  #        instantiates any optional argument and then delegates to s
  #        CUDA C function (not C++).  These wrappers are defining the
  #        C symbols, and not the name-mangled C++ symbols.
  fnc_signature = (fnc["name"] + '(' +
               ', '.join([arg["type"] + ' ' + arg["name"]
                          for arg in args]) +
               ')'
              )
  fnc_args = (''.join(["  " + arg["type"] + ' ' + arg["name"] + ";\n"
                      for arg in args])
             )
  fnc_call = (fnc["name"] + '(' +
               ', '.join([arg["name"] for arg in args]) +
               ')'
             )
  # These parameters must be passed to proxy as copy-by-value
  in_style_tags = ["IN", "SIZE", "DEST", "SRC",
                   "DEST_PITCH", "SRC_PITCH", "HEIGHT", "DIRECTION"]
  direction_declared = False;
  flushDirtyPages_prolog = (
  """// TODO: Ideally, we should flush only when the function uses the
  // data from the managed regions
  if (haveDirtyPages)
    flushDirtyPages();""")

  cudawrappers_prolog = (
"""{
  if (!initialized)
    proxy_initialize();

  %s

  %s ret_val;
  char send_buf[1000];
  char recv_buf[1000];
  int chars_sent = 0;
  int chars_rcvd = 0;

""" % (flushDirtyPages_prolog if fnc["flushDirtyPages"] else "", fnc["type"].replace("EXTERNC ", "")))

  if [myarg for myarg in args if myarg["tag"][0] == "DIRECTION"]:
    for myarg in args:
      if myarg["tag"][0] == "DEST":
        dest_var = myarg["name"]
      elif myarg["tag"][0] == "SRC":
        src_var = myarg["name"]
      elif myarg["tag"][0] == "DIRECTION":
        direction_var = myarg["name"]
    cudawrappers_prolog += (
"""  enum cudaMemcpyKind _direction = kind;
  if (%s == cudaMemcpyDefault) {
    cudaMemcpyGetDirection(%s, %s, &_direction);
    %s = _direction;
  }

""" % (direction_var, dest_var, src_var, direction_var))

  cudawrappers_epilog = (
"""
  log_append(strce_to_send);

  return ret_val;
}

""")

  cudaproxy_prolog = (
"""  char send_buf[1000];
  int chars_sent = 0;
""")
  def some_incoming_args(args):
    return [arg for arg in args if arg["tag"][0] not in ["OUT"]]
  if some_incoming_args(args):
    cudaproxy_prolog += (
"""  char recv_buf[1000];
  int chars_rcvd = 0;
""")

  #====
  # Now write body of application function and proxy function

  # This code process the messages due to cudaMemcpy as extra messages.
  # It is inserted after sending and receiving messages in application and proxy
  (application_before, application_after, proxy_before, proxy_after) = \
    cudaMemcpyExtraCode(args, fnc["isLogging"])

  cuda_include.write("  OP_" + fnc["name"] + ",\n")
  cuda_include2.write("void FNC_" + fnc["name"] + "(void);\n")

  cudawrappers.write(fnc["type"] + "\n" + fnc_signature + "\n")
  if fnc["type"].startswith("EXTERNC "):
    # Remove "EXTERNC for all uses after this (inside a function)
    fnc["type"] = fnc["type"][len("EXTERNC "):]

  cudawrappers.write(cudawrappers_prolog)
  cudawrappers.write(
"""  // Write the IN arguments (and INOUT and IN_DEEPCOPY) to the proxy
  enum cuda_op op = OP_%s;
  memcpy(send_buf + chars_sent, &op, sizeof op);
  chars_sent += sizeof(enum cuda_op);
""" % fnc["name"])
  for arg in args:
    if arg["tag"][0] in in_style_tags:
      (var, size) = (arg["name"], "sizeof " + arg["name"])
      cudawrappers.write(("  memcpy(send_buf + chars_sent, & %s, %s);\n" +
                          "  chars_sent += %s;\n") % (var, size, size))
    elif arg["tag"][0] in ["IN_DEEPCOPY", "INOUT"]:
      (var, size) = (arg["name"], "sizeof *" + arg["name"])
      cudawrappers.write(("  memcpy(send_buf + chars_sent, %s, %s);\n" +
                          "  chars_sent += %s;\n") % (var, size, size))

  cudawrappers.write(
"""
  // Send op code and args to proxy
  JASSERT(write(skt_master, send_buf, chars_sent) == chars_sent)
         (JASSERT_ERRNO);
""")
  # NOTE:  We should log CUDA fnc's only with prim. args; e.g., not cudaMemcpy()
  if (fnc["isLogging"]):
    cudawrappers.write(
"""  log_append(send_buf, chars_sent);
""")
  # This occurs before we send to the proxy process because
  #   application_before does not use send_buf.  It does its own send,
  #   since this is typically a pointer to a buffer in the application code.
  cudawrappers.write(application_before)

  sizeof_args = [" + sizeof *" + arg["name"] for arg in args
                                      if arg["tag"][0] in ["OUT", "INOUT"]]
  sizeof_args = ''.join(sizeof_args)
  if sizeof_args.startswith(" + "):
    sizeof_args = sizeof_args[len(" + "):]
  cudawrappers.write(
"""
  // Receive the OUT arguments after the proxy made the function call
  // Compute total chars_rcvd to be read in the next msg
""")
  if len(sizeof_args) > 0:
    cudawrappers.write("  chars_rcvd = %s + sizeof ret_val;\n" % sizeof_args)
  else:
    cudawrappers.write("  // (No primitive args to receive, except ret_val.)\n")
    cudawrappers.write("  chars_rcvd = sizeof ret_val;\n")
  cudawrappers.write(
"""  JASSERT(read(skt_master, recv_buf, chars_rcvd) == chars_rcvd)
         (JASSERT_ERRNO);

  // Extract OUT variables
  chars_rcvd = 0;
""")
  for arg in args:
    if arg["tag"][0] in ["OUT", "INOUT"]:
      (var, size) = (arg["name"], "sizeof *" + arg["name"])
      # Strip one '*' from arg["type"] on next line
      assert arg["type"].rstrip().endswith('*')
      cudawrappers.write(("  memcpy(%s, recv_buf + chars_rcvd, %s);\n" +
                          "  chars_rcvd += %s;\n") % (var, size, size))
  cudawrappers.write(
"""
  memcpy(&ret_val, recv_buf + chars_rcvd, sizeof ret_val);
""")

  # This occurs after we send to the proxy process because
  #   application_after does not use recv_buf.  It does its own recv, since
  #   the sender typically sent a pointer to a buffer in the application code.
  cudawrappers.write(application_after)

  cudawrappers.write("""
  return ret_val;
}

""")

  cudaproxy.write("    case OP_" + fnc["name"] + ":\n")
  cudaproxy.write("      FNC_" + fnc["name"] + "();\n      break;\n")

  # Write FNC_XXX() declarations into the second half of the .h file.
  cudaproxy2.write("void FNC_" + fnc["name"] + "(void) {\n")
  cudaproxy2.write(fnc_args.replace("const ", "") + "\n")
  cudaproxy2.write(cudaproxy_prolog)

  args_in_sizeof = [" + sizeof " + arg["name"] for arg in args
                                             if arg["tag"][0] in in_style_tags]
  args_in_sizeof += [" + sizeof *" + arg["name"] for arg in args
                                if arg["tag"][0] in ["IN_DEEPCOPY", "INOUT"]]
  args_in_sizeof = ''.join(args_in_sizeof)
  if len(args_in_sizeof) >= 3:
    args_in_sizeof = args_in_sizeof[3:]  # Remove initial " + "
  cudaproxy2.write(
"""
  // Receive the arguments
""" +
  # Python inline if-else:
  (
"""  // Compute total chars_rcvd to be read in the next msg
  chars_rcvd = """ + args_in_sizeof + """;
  assert(read(skt_accept, recv_buf, chars_rcvd) == chars_rcvd);
  // Now read the data for the total chars_rcvd
  chars_rcvd = 0;
"""
  if len(args_in_sizeof) > 0
  else "  // No primitive args to receive.  Will not read from skt_accept.\n")
)
  for arg in args:
    if arg["tag"][0] in in_style_tags:  # if copy-by-value parameter
      (var, size) = (arg["name"], "sizeof " + arg["name"])
      cudaproxy2.write(("  memcpy(&%s, recv_buf + chars_rcvd, %s);\n" +
                        "  chars_rcvd += %s;\n") % (var, size, size))
  for arg in args:
    if arg["tag"][0] in in_style_tags:
      pass  # Already handled above
    elif arg["tag"][0] in ["IN_DEEPCOPY", "INOUT"]:
      # Typically, the cuda parameter is of type:  struct cudaSomething *param
      # Strip one '*' from arg["type"] on next line
      (type, base_var, size) = (arg["type"].rstrip()[:-1].rstrip(),
                                "base_" + arg["name"], "sizeof *" + arg["name"])
      assert arg["type"].rstrip()[-1] == '*'
      cudaproxy2.write(
"""  // Declare base variables for IN_DEEPCOPY and INOUT arguments to point to
  %s %s;
  %s = &%s;
  memcpy(&%s, recv_buf + chars_rcvd, %s);
  chars_rcvd += %s;
""" % (type.replace("const ", ""), base_var, arg["name"], base_var, base_var,
       size, size))

  # This occurs after we receive from the application process because
  #   application_before did not use the send_buf.  It sent its own send,
  #   since this was typically a pointer to a buffer in the application code.
  cudaproxy2.write(proxy_before)

  args_out = [arg for arg in args if arg["tag"][0] in ["OUT", "INOUT"]]
  # FIXME:  This "// Declare base variables" seems to be repeated. Remove???
  if len(args_out) > 0:
    cudaproxy2.write("""
  // Declare base variables for OUT arguments to point to
""")
  for arg in args_out:
    if arg["tag"][0] in ["OUT", "INOUT"]:
      # Strip one '*' from arg["type"] on next line
      (type, base_var) = (arg["type"].rstrip()[:-1].rstrip(),
                          "base_" + arg["name"])
      assert arg["type"].rstrip()[-1] == '*'
      if arg["tag"][0] in ["OUT"]:
        cudaproxy2.write(("  %s %s;\n") % (type, base_var))
      cudaproxy2.write(("  %s = &%s;\n") % (arg["name"], base_var))

  cudaproxy2.write(
"""
  // Make the function call
""")
  cudaproxy2.write("  " + fnc["type"] + " ret_val = " + fnc_call + ";\n")

  cudaproxy2.write("""
  // Write back the arguments to the application
""")
  for arg in args:
    if arg["tag"][0] in ["OUT", "INOUT"]:
      (var, size) = (arg["name"], "sizeof *" + arg["name"])
      cudaproxy2.write(("  memcpy(send_buf + chars_sent, %s, %s);\n" +
                        "  chars_sent += %s;\n") % (var, size, size))
  cudaproxy2.write("""  memcpy(send_buf + chars_sent, &ret_val, sizeof ret_val);
  chars_sent += sizeof ret_val;
""")
  cudaproxy2.write(
"""  assert(write(skt_accept, send_buf, chars_sent) == chars_sent);
""")
  # This occurs after we send to the application process, because
  #   application_after is not using the send_buf.  It does its own send, since
  #   the application process typically uses a pointer to a buffer
  #   in the application code for this large buffer.
  cudaproxy2.write(proxy_after)
  # End of function.  No 'return' stmt for fnc returning void.
  cudaproxy2.write(
"""};

""")


# ====
# Back to top level.  write into files and collect all of them together.

# Write body of each file, with wrapper function, proxy stub fnc, .h fnc decl.
for ast_annotated in ast_annotated_wrappers:
  write_all_files(ast_annotated)

# Write end of each file and close
cuda_include.write(cuda_include_tail)
cuda_include.write("\n\n")
cuda_include2.seek(0)
cuda_include.write(cuda_include2.read())
cuda_include.write("\n#endif // ifndef _CUDA_PLUGIN_H_")
cuda_include2.close()
os.remove(cuda_include2.name)
cuda_include.close()

cudawrappers.write(cudawrappers_tail)
cudawrappers.write(verbatim_wrappers)
cudawrappers.close()

cudaproxy.write(cudaproxy_tail)
cudaproxy.write("\n\n")
cudaproxy.write(verbatim_proxies)
cudaproxy2.seek(0)
cudaproxy.write(cudaproxy2.read())
cudaproxy2.close()
os.remove(cudaproxy2.name)
cudaproxy.close()


print "Done"
