��&
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BatchToSpaceND

input"T
block_shape"Tblock_shape
crops"Tcrops
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
Tcropstype0:
2	
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
SpaceToBatchND

input"T
block_shape"Tblock_shape
paddings"	Tpaddings
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
	Tpaddingstype0:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��!
�
8cond_1/Adam/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*I
shared_name:8cond_1/Adam/multi_head_attention/attention_output/bias/v
�
Lcond_1/Adam/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp8cond_1/Adam/multi_head_attention/attention_output/bias/v*
_output_shapes	
:�*
dtype0
�
:cond_1/Adam/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*K
shared_name<:cond_1/Adam/multi_head_attention/attention_output/kernel/v
�
Ncond_1/Adam/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp:cond_1/Adam/multi_head_attention/attention_output/kernel/v*#
_output_shapes
: �*
dtype0
�
-cond_1/Adam/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *>
shared_name/-cond_1/Adam/multi_head_attention/value/bias/v
�
Acond_1/Adam/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp-cond_1/Adam/multi_head_attention/value/bias/v*
_output_shapes

: *
dtype0
�
/cond_1/Adam/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *@
shared_name1/cond_1/Adam/multi_head_attention/value/kernel/v
�
Ccond_1/Adam/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp/cond_1/Adam/multi_head_attention/value/kernel/v*#
_output_shapes
:� *
dtype0
�
+cond_1/Adam/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *<
shared_name-+cond_1/Adam/multi_head_attention/key/bias/v
�
?cond_1/Adam/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp+cond_1/Adam/multi_head_attention/key/bias/v*
_output_shapes

: *
dtype0
�
-cond_1/Adam/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *>
shared_name/-cond_1/Adam/multi_head_attention/key/kernel/v
�
Acond_1/Adam/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp-cond_1/Adam/multi_head_attention/key/kernel/v*#
_output_shapes
:� *
dtype0
�
-cond_1/Adam/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *>
shared_name/-cond_1/Adam/multi_head_attention/query/bias/v
�
Acond_1/Adam/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp-cond_1/Adam/multi_head_attention/query/bias/v*
_output_shapes

: *
dtype0
�
/cond_1/Adam/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *@
shared_name1/cond_1/Adam/multi_head_attention/query/kernel/v
�
Ccond_1/Adam/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp/cond_1/Adam/multi_head_attention/query/kernel/v*#
_output_shapes
:� *
dtype0
�
cond_1/Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/dense/bias/v
�
,cond_1/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/bias/v*
_output_shapes
:*
dtype0
�
cond_1/Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_namecond_1/Adam/dense/kernel/v
�
.cond_1/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/kernel/v*
_output_shapes
:	�*
dtype0
�
cond_1/Adam/conv1d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_7/bias/v
�
/cond_1/Adam/conv1d_7/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_7/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_namecond_1/Adam/conv1d_7/kernel/v
�
1cond_1/Adam/conv1d_7/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_7/kernel/v*$
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv1d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_6/bias/v
�
/cond_1/Adam/conv1d_6/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_6/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_namecond_1/Adam/conv1d_6/kernel/v
�
1cond_1/Adam/conv1d_6/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_6/kernel/v*$
_output_shapes
:��*
dtype0
�
(cond_1/Adam/layer_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(cond_1/Adam/layer_normalization_2/beta/v
�
<cond_1/Adam/layer_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp(cond_1/Adam/layer_normalization_2/beta/v*
_output_shapes	
:�*
dtype0
�
)cond_1/Adam/layer_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*:
shared_name+)cond_1/Adam/layer_normalization_2/gamma/v
�
=cond_1/Adam/layer_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/layer_normalization_2/gamma/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_5/bias/v
�
/cond_1/Adam/conv1d_5/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_5/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*.
shared_namecond_1/Adam/conv1d_5/kernel/v
�
1cond_1/Adam/conv1d_5/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_5/kernel/v*#
_output_shapes
:@�*
dtype0
�
cond_1/Adam/conv1d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_4/bias/v
�
/cond_1/Adam/conv1d_4/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_4/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_namecond_1/Adam/conv1d_4/kernel/v
�
1cond_1/Adam/conv1d_4/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_4/kernel/v*$
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_3/bias/v
�
/cond_1/Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_3/bias/v*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*.
shared_namecond_1/Adam/conv1d_3/kernel/v
�
1cond_1/Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_3/kernel/v*#
_output_shapes
:@�*
dtype0
�
(cond_1/Adam/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(cond_1/Adam/layer_normalization_1/beta/v
�
<cond_1/Adam/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp(cond_1/Adam/layer_normalization_1/beta/v*
_output_shapes
:@*
dtype0
�
)cond_1/Adam/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/layer_normalization_1/gamma/v
�
=cond_1/Adam/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp)cond_1/Adam/layer_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
�
cond_1/Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv1d_2/bias/v
�
/cond_1/Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_2/bias/v*
_output_shapes
:@*
dtype0
�
cond_1/Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@*.
shared_namecond_1/Adam/conv1d_2/kernel/v
�
1cond_1/Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_2/kernel/v*"
_output_shapes
:`@*
dtype0
�
cond_1/Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv1d_1/bias/v
�
/cond_1/Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_1/bias/v*
_output_shapes
:@*
dtype0
�
cond_1/Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_namecond_1/Adam/conv1d_1/kernel/v
�
1cond_1/Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_1/kernel/v*"
_output_shapes
:@@*
dtype0
�
cond_1/Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namecond_1/Adam/conv1d/bias/v
�
-cond_1/Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d/bias/v*
_output_shapes
:@*
dtype0
�
cond_1/Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@*,
shared_namecond_1/Adam/conv1d/kernel/v
�
/cond_1/Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d/kernel/v*"
_output_shapes
:`@*
dtype0
�
&cond_1/Adam/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&cond_1/Adam/layer_normalization/beta/v
�
:cond_1/Adam/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp&cond_1/Adam/layer_normalization/beta/v*
_output_shapes
:`*
dtype0
�
'cond_1/Adam/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*8
shared_name)'cond_1/Adam/layer_normalization/gamma/v
�
;cond_1/Adam/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp'cond_1/Adam/layer_normalization/gamma/v*
_output_shapes
:`*
dtype0
�
8cond_1/Adam/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*I
shared_name:8cond_1/Adam/multi_head_attention/attention_output/bias/m
�
Lcond_1/Adam/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp8cond_1/Adam/multi_head_attention/attention_output/bias/m*
_output_shapes	
:�*
dtype0
�
:cond_1/Adam/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*K
shared_name<:cond_1/Adam/multi_head_attention/attention_output/kernel/m
�
Ncond_1/Adam/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp:cond_1/Adam/multi_head_attention/attention_output/kernel/m*#
_output_shapes
: �*
dtype0
�
-cond_1/Adam/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *>
shared_name/-cond_1/Adam/multi_head_attention/value/bias/m
�
Acond_1/Adam/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp-cond_1/Adam/multi_head_attention/value/bias/m*
_output_shapes

: *
dtype0
�
/cond_1/Adam/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *@
shared_name1/cond_1/Adam/multi_head_attention/value/kernel/m
�
Ccond_1/Adam/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp/cond_1/Adam/multi_head_attention/value/kernel/m*#
_output_shapes
:� *
dtype0
�
+cond_1/Adam/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *<
shared_name-+cond_1/Adam/multi_head_attention/key/bias/m
�
?cond_1/Adam/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp+cond_1/Adam/multi_head_attention/key/bias/m*
_output_shapes

: *
dtype0
�
-cond_1/Adam/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *>
shared_name/-cond_1/Adam/multi_head_attention/key/kernel/m
�
Acond_1/Adam/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp-cond_1/Adam/multi_head_attention/key/kernel/m*#
_output_shapes
:� *
dtype0
�
-cond_1/Adam/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *>
shared_name/-cond_1/Adam/multi_head_attention/query/bias/m
�
Acond_1/Adam/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp-cond_1/Adam/multi_head_attention/query/bias/m*
_output_shapes

: *
dtype0
�
/cond_1/Adam/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *@
shared_name1/cond_1/Adam/multi_head_attention/query/kernel/m
�
Ccond_1/Adam/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp/cond_1/Adam/multi_head_attention/query/kernel/m*#
_output_shapes
:� *
dtype0
�
cond_1/Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/dense/bias/m
�
,cond_1/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/bias/m*
_output_shapes
:*
dtype0
�
cond_1/Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_namecond_1/Adam/dense/kernel/m
�
.cond_1/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/kernel/m*
_output_shapes
:	�*
dtype0
�
cond_1/Adam/conv1d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_7/bias/m
�
/cond_1/Adam/conv1d_7/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_7/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_namecond_1/Adam/conv1d_7/kernel/m
�
1cond_1/Adam/conv1d_7/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_7/kernel/m*$
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv1d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_6/bias/m
�
/cond_1/Adam/conv1d_6/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_6/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_namecond_1/Adam/conv1d_6/kernel/m
�
1cond_1/Adam/conv1d_6/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_6/kernel/m*$
_output_shapes
:��*
dtype0
�
(cond_1/Adam/layer_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(cond_1/Adam/layer_normalization_2/beta/m
�
<cond_1/Adam/layer_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp(cond_1/Adam/layer_normalization_2/beta/m*
_output_shapes	
:�*
dtype0
�
)cond_1/Adam/layer_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*:
shared_name+)cond_1/Adam/layer_normalization_2/gamma/m
�
=cond_1/Adam/layer_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/layer_normalization_2/gamma/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_5/bias/m
�
/cond_1/Adam/conv1d_5/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_5/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*.
shared_namecond_1/Adam/conv1d_5/kernel/m
�
1cond_1/Adam/conv1d_5/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_5/kernel/m*#
_output_shapes
:@�*
dtype0
�
cond_1/Adam/conv1d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_4/bias/m
�
/cond_1/Adam/conv1d_4/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_4/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_namecond_1/Adam/conv1d_4/kernel/m
�
1cond_1/Adam/conv1d_4/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_4/kernel/m*$
_output_shapes
:��*
dtype0
�
cond_1/Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namecond_1/Adam/conv1d_3/bias/m
�
/cond_1/Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_3/bias/m*
_output_shapes	
:�*
dtype0
�
cond_1/Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*.
shared_namecond_1/Adam/conv1d_3/kernel/m
�
1cond_1/Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_3/kernel/m*#
_output_shapes
:@�*
dtype0
�
(cond_1/Adam/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(cond_1/Adam/layer_normalization_1/beta/m
�
<cond_1/Adam/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp(cond_1/Adam/layer_normalization_1/beta/m*
_output_shapes
:@*
dtype0
�
)cond_1/Adam/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cond_1/Adam/layer_normalization_1/gamma/m
�
=cond_1/Adam/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp)cond_1/Adam/layer_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
�
cond_1/Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv1d_2/bias/m
�
/cond_1/Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_2/bias/m*
_output_shapes
:@*
dtype0
�
cond_1/Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@*.
shared_namecond_1/Adam/conv1d_2/kernel/m
�
1cond_1/Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_2/kernel/m*"
_output_shapes
:`@*
dtype0
�
cond_1/Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv1d_1/bias/m
�
/cond_1/Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_1/bias/m*
_output_shapes
:@*
dtype0
�
cond_1/Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_namecond_1/Adam/conv1d_1/kernel/m
�
1cond_1/Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_1/kernel/m*"
_output_shapes
:@@*
dtype0
�
cond_1/Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namecond_1/Adam/conv1d/bias/m
�
-cond_1/Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d/bias/m*
_output_shapes
:@*
dtype0
�
cond_1/Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@*,
shared_namecond_1/Adam/conv1d/kernel/m
�
/cond_1/Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d/kernel/m*"
_output_shapes
:`@*
dtype0
�
&cond_1/Adam/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&cond_1/Adam/layer_normalization/beta/m
�
:cond_1/Adam/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp&cond_1/Adam/layer_normalization/beta/m*
_output_shapes
:`*
dtype0
�
'cond_1/Adam/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*8
shared_name)'cond_1/Adam/layer_normalization/gamma/m
�
;cond_1/Adam/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp'cond_1/Adam/layer_normalization/gamma/m*
_output_shapes
:`*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
z
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_positives_1
s
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
:*
dtype0
x
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_2
q
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
h

good_stepsVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
good_steps
a
good_steps/Read/ReadVariableOpReadVariableOp
good_steps*
_output_shapes
: *
dtype0	
x
current_loss_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecurrent_loss_scale
q
&current_loss_scale/Read/ReadVariableOpReadVariableOpcurrent_loss_scale*
_output_shapes
: *
dtype0
�
cond_1/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namecond_1/Adam/learning_rate

-cond_1/Adam/learning_rate/Read/ReadVariableOpReadVariableOpcond_1/Adam/learning_rate*
_output_shapes
: *
dtype0
v
cond_1/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namecond_1/Adam/decay
o
%cond_1/Adam/decay/Read/ReadVariableOpReadVariableOpcond_1/Adam/decay*
_output_shapes
: *
dtype0
x
cond_1/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_2
q
&cond_1/Adam/beta_2/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_2*
_output_shapes
: *
dtype0
x
cond_1/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_1
q
&cond_1/Adam/beta_1/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_1*
_output_shapes
: *
dtype0
t
cond_1/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *!
shared_namecond_1/Adam/iter
m
$cond_1/Adam/iter/Read/ReadVariableOpReadVariableOpcond_1/Adam/iter*
_output_shapes
: *
dtype0	
�
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*multi_head_attention/attention_output/bias
�
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*=
shared_name.,multi_head_attention/attention_output/kernel
�
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*#
_output_shapes
: �*
dtype0
�
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!multi_head_attention/value/bias
�
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

: *
dtype0
�
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *2
shared_name#!multi_head_attention/value/kernel
�
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*#
_output_shapes
:� *
dtype0
�
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_namemulti_head_attention/key/bias
�
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

: *
dtype0
�
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *0
shared_name!multi_head_attention/key/kernel
�
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*#
_output_shapes
:� *
dtype0
�
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!multi_head_attention/query/bias
�
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

: *
dtype0
�
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *2
shared_name#!multi_head_attention/query/kernel
�
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*#
_output_shapes
:� *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
s
conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1d_7/bias
l
!conv1d_7/bias/Read/ReadVariableOpReadVariableOpconv1d_7/bias*
_output_shapes	
:�*
dtype0
�
conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv1d_7/kernel
y
#conv1d_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_7/kernel*$
_output_shapes
:��*
dtype0
s
conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1d_6/bias
l
!conv1d_6/bias/Read/ReadVariableOpReadVariableOpconv1d_6/bias*
_output_shapes	
:�*
dtype0
�
conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv1d_6/kernel
y
#conv1d_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_6/kernel*$
_output_shapes
:��*
dtype0
�
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namelayer_normalization_2/beta
�
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namelayer_normalization_2/gamma
�
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes	
:�*
dtype0
s
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1d_5/bias
l
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes	
:�*
dtype0

conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv1d_5/kernel
x
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*#
_output_shapes
:@�*
dtype0
s
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1d_4/bias
l
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes	
:�*
dtype0
�
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv1d_4/kernel
y
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*$
_output_shapes
:��*
dtype0
s
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1d_3/bias
l
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes	
:�*
dtype0

conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv1d_3/kernel
x
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*#
_output_shapes
:@�*
dtype0
�
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_1/beta
�
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:@*
dtype0
�
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_1/gamma
�
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:@*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:@*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:`@*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:@@*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:`@*
dtype0
�
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*)
shared_namelayer_normalization/beta
�
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:`*
dtype0
�
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`**
shared_namelayer_normalization/gamma
�
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:`*
dtype0
�
serving_default_input_1Placeholder*+
_output_shapes
:���������`*
dtype0* 
shape:���������`
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer_normalization/gammalayer_normalization/betaconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biaslayer_normalization_1/gammalayer_normalization_1/betaconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biaslayer_normalization_2/gammalayer_normalization_2/betaconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biasdense/kernel
dense/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_1973951

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B߃
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer-19
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer_with_weights-12
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures*
* 
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)axis
	*gamma
+beta*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_random_generator* 
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias
 D_jit_compiled_convolution_op*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator* 

U	keras_api* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\axis
	]gamma
^beta*
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias
 g_jit_compiled_convolution_op*
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
n_random_generator* 
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias
 w_jit_compiled_convolution_op*
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
*0
+1
22
33
B4
C5
K6
L7
]8
^9
e10
f11
u12
v13
~14
15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31*
�
*0
+1
22
33
B4
C5
K6
L7
]8
^9
e10
f11
u12
v13
~14
15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
loss_scale
�base_optimizer
	�iter
�beta_1
�beta_2

�decay
�learning_rate*m�+m�2m�3m�Bm�Cm�Km�Lm�]m�^m�em�fm�um�vm�~m�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�*v�+v�2v�3v�Bv�Cv�Kv�Lv�]v�^v�ev�fv�uv�vv�~v�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*

�serving_default* 

*0
+1*

*0
+1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

B0
C1*

B0
C1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

K0
L1*

K0
L1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 

]0
^1*

]0
^1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

e0
f1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

u0
v1*

u0
v1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv1d_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

~0
1*

~0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv1d_5/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_5/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv1d_6/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_6/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv1d_7/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_7/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention/query/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/query/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/key/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEmulti_head_attention/key/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention/value/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/value/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention/attention_output/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*multi_head_attention/attention_output/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24*
,
�0
�1
�2
�3
�4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
.
�current_loss_scale
�
good_steps*
* 
SM
VARIABLE_VALUEcond_1/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEcond_1/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEcond_1/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcond_1/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEcond_1/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
`
�	variables
�	keras_api
�
thresholds
�true_positives
�false_positives*
`
�	variables
�	keras_api
�
thresholds
�true_positives
�false_negatives*
z
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives*
nh
VARIABLE_VALUEcurrent_loss_scaleBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
good_steps:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
* 
ga
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE'cond_1/Adam/layer_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE&cond_1/Adam/layer_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEcond_1/Adam/conv1d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE)cond_1/Adam/layer_normalization_1/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(cond_1/Adam/layer_normalization_1/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_4/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_4/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_5/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_5/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE)cond_1/Adam/layer_normalization_2/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(cond_1/Adam/layer_normalization_2/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_6/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_6/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_7/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_7/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/dense/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEcond_1/Adam/dense/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/cond_1/Adam/multi_head_attention/query/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-cond_1/Adam/multi_head_attention/query/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-cond_1/Adam/multi_head_attention/key/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE+cond_1/Adam/multi_head_attention/key/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/cond_1/Adam/multi_head_attention/value/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-cond_1/Adam/multi_head_attention/value/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:cond_1/Adam/multi_head_attention/attention_output/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE8cond_1/Adam/multi_head_attention/attention_output/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE'cond_1/Adam/layer_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE&cond_1/Adam/layer_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEcond_1/Adam/conv1d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE)cond_1/Adam/layer_normalization_1/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(cond_1/Adam/layer_normalization_1/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_4/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_4/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_5/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_5/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE)cond_1/Adam/layer_normalization_2/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE(cond_1/Adam/layer_normalization_2/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_6/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEcond_1/Adam/conv1d_6/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_7/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/conv1d_7/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEcond_1/Adam/dense/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEcond_1/Adam/dense/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/cond_1/Adam/multi_head_attention/query/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-cond_1/Adam/multi_head_attention/query/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-cond_1/Adam/multi_head_attention/key/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE+cond_1/Adam/multi_head_attention/key/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/cond_1/Adam/multi_head_attention/value/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-cond_1/Adam/multi_head_attention/value/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:cond_1/Adam/multi_head_attention/attention_output/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE8cond_1/Adam/multi_head_attention/attention_output/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp#conv1d_6/kernel/Read/ReadVariableOp!conv1d_6/bias/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp!conv1d_7/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOp$cond_1/Adam/iter/Read/ReadVariableOp&cond_1/Adam/beta_1/Read/ReadVariableOp&cond_1/Adam/beta_2/Read/ReadVariableOp%cond_1/Adam/decay/Read/ReadVariableOp-cond_1/Adam/learning_rate/Read/ReadVariableOp&current_loss_scale/Read/ReadVariableOpgood_steps/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp;cond_1/Adam/layer_normalization/gamma/m/Read/ReadVariableOp:cond_1/Adam/layer_normalization/beta/m/Read/ReadVariableOp/cond_1/Adam/conv1d/kernel/m/Read/ReadVariableOp-cond_1/Adam/conv1d/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_1/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_1/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_2/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_2/bias/m/Read/ReadVariableOp=cond_1/Adam/layer_normalization_1/gamma/m/Read/ReadVariableOp<cond_1/Adam/layer_normalization_1/beta/m/Read/ReadVariableOp1cond_1/Adam/conv1d_3/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_3/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_4/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_4/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_5/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_5/bias/m/Read/ReadVariableOp=cond_1/Adam/layer_normalization_2/gamma/m/Read/ReadVariableOp<cond_1/Adam/layer_normalization_2/beta/m/Read/ReadVariableOp1cond_1/Adam/conv1d_6/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_6/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_7/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_7/bias/m/Read/ReadVariableOp.cond_1/Adam/dense/kernel/m/Read/ReadVariableOp,cond_1/Adam/dense/bias/m/Read/ReadVariableOpCcond_1/Adam/multi_head_attention/query/kernel/m/Read/ReadVariableOpAcond_1/Adam/multi_head_attention/query/bias/m/Read/ReadVariableOpAcond_1/Adam/multi_head_attention/key/kernel/m/Read/ReadVariableOp?cond_1/Adam/multi_head_attention/key/bias/m/Read/ReadVariableOpCcond_1/Adam/multi_head_attention/value/kernel/m/Read/ReadVariableOpAcond_1/Adam/multi_head_attention/value/bias/m/Read/ReadVariableOpNcond_1/Adam/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpLcond_1/Adam/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp;cond_1/Adam/layer_normalization/gamma/v/Read/ReadVariableOp:cond_1/Adam/layer_normalization/beta/v/Read/ReadVariableOp/cond_1/Adam/conv1d/kernel/v/Read/ReadVariableOp-cond_1/Adam/conv1d/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_1/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_1/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_2/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_2/bias/v/Read/ReadVariableOp=cond_1/Adam/layer_normalization_1/gamma/v/Read/ReadVariableOp<cond_1/Adam/layer_normalization_1/beta/v/Read/ReadVariableOp1cond_1/Adam/conv1d_3/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_3/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_4/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_4/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_5/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_5/bias/v/Read/ReadVariableOp=cond_1/Adam/layer_normalization_2/gamma/v/Read/ReadVariableOp<cond_1/Adam/layer_normalization_2/beta/v/Read/ReadVariableOp1cond_1/Adam/conv1d_6/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_6/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_7/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_7/bias/v/Read/ReadVariableOp.cond_1/Adam/dense/kernel/v/Read/ReadVariableOp,cond_1/Adam/dense/bias/v/Read/ReadVariableOpCcond_1/Adam/multi_head_attention/query/kernel/v/Read/ReadVariableOpAcond_1/Adam/multi_head_attention/query/bias/v/Read/ReadVariableOpAcond_1/Adam/multi_head_attention/key/kernel/v/Read/ReadVariableOp?cond_1/Adam/multi_head_attention/key/bias/v/Read/ReadVariableOpCcond_1/Adam/multi_head_attention/value/kernel/v/Read/ReadVariableOpAcond_1/Adam/multi_head_attention/value/bias/v/Read/ReadVariableOpNcond_1/Adam/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpLcond_1/Adam/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpConst*�
Tiny
w2u		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *)
f$R"
 __inference__traced_save_1975890
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betaconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biaslayer_normalization_1/gammalayer_normalization_1/betaconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biaslayer_normalization_2/gammalayer_normalization_2/betaconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasdense/kernel
dense/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biascond_1/Adam/itercond_1/Adam/beta_1cond_1/Adam/beta_2cond_1/Adam/decaycond_1/Adam/learning_ratecurrent_loss_scale
good_stepstotal_1count_1totalcounttrue_positives_2false_positives_1true_positives_1false_negatives_1true_positivestrue_negativesfalse_positivesfalse_negatives'cond_1/Adam/layer_normalization/gamma/m&cond_1/Adam/layer_normalization/beta/mcond_1/Adam/conv1d/kernel/mcond_1/Adam/conv1d/bias/mcond_1/Adam/conv1d_1/kernel/mcond_1/Adam/conv1d_1/bias/mcond_1/Adam/conv1d_2/kernel/mcond_1/Adam/conv1d_2/bias/m)cond_1/Adam/layer_normalization_1/gamma/m(cond_1/Adam/layer_normalization_1/beta/mcond_1/Adam/conv1d_3/kernel/mcond_1/Adam/conv1d_3/bias/mcond_1/Adam/conv1d_4/kernel/mcond_1/Adam/conv1d_4/bias/mcond_1/Adam/conv1d_5/kernel/mcond_1/Adam/conv1d_5/bias/m)cond_1/Adam/layer_normalization_2/gamma/m(cond_1/Adam/layer_normalization_2/beta/mcond_1/Adam/conv1d_6/kernel/mcond_1/Adam/conv1d_6/bias/mcond_1/Adam/conv1d_7/kernel/mcond_1/Adam/conv1d_7/bias/mcond_1/Adam/dense/kernel/mcond_1/Adam/dense/bias/m/cond_1/Adam/multi_head_attention/query/kernel/m-cond_1/Adam/multi_head_attention/query/bias/m-cond_1/Adam/multi_head_attention/key/kernel/m+cond_1/Adam/multi_head_attention/key/bias/m/cond_1/Adam/multi_head_attention/value/kernel/m-cond_1/Adam/multi_head_attention/value/bias/m:cond_1/Adam/multi_head_attention/attention_output/kernel/m8cond_1/Adam/multi_head_attention/attention_output/bias/m'cond_1/Adam/layer_normalization/gamma/v&cond_1/Adam/layer_normalization/beta/vcond_1/Adam/conv1d/kernel/vcond_1/Adam/conv1d/bias/vcond_1/Adam/conv1d_1/kernel/vcond_1/Adam/conv1d_1/bias/vcond_1/Adam/conv1d_2/kernel/vcond_1/Adam/conv1d_2/bias/v)cond_1/Adam/layer_normalization_1/gamma/v(cond_1/Adam/layer_normalization_1/beta/vcond_1/Adam/conv1d_3/kernel/vcond_1/Adam/conv1d_3/bias/vcond_1/Adam/conv1d_4/kernel/vcond_1/Adam/conv1d_4/bias/vcond_1/Adam/conv1d_5/kernel/vcond_1/Adam/conv1d_5/bias/v)cond_1/Adam/layer_normalization_2/gamma/v(cond_1/Adam/layer_normalization_2/beta/vcond_1/Adam/conv1d_6/kernel/vcond_1/Adam/conv1d_6/bias/vcond_1/Adam/conv1d_7/kernel/vcond_1/Adam/conv1d_7/bias/vcond_1/Adam/dense/kernel/vcond_1/Adam/dense/bias/v/cond_1/Adam/multi_head_attention/query/kernel/v-cond_1/Adam/multi_head_attention/query/bias/v-cond_1/Adam/multi_head_attention/key/kernel/v+cond_1/Adam/multi_head_attention/key/bias/v/cond_1/Adam/multi_head_attention/value/kernel/v-cond_1/Adam/multi_head_attention/value/bias/v:cond_1/Adam/multi_head_attention/attention_output/kernel/v8cond_1/Adam/multi_head_attention/attention_output/bias/v*
Tinx
v2t*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__traced_restore_1976245��
�
�
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1972900

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       `
PadPadinputsPad/paddings:output:0*
T0*,
_output_shapes
:����������^
Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:}
3Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
5Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
0Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
-Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       k
!Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:w
Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
Conv1D/SpaceToBatchNDSpaceToBatchNDPad:output:0*Conv1D/SpaceToBatchND/block_shape:output:0'Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsConv1D/SpaceToBatchND:output:0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������k
!Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:t
Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
Conv1D/BatchToSpaceNDBatchToSpaceNDConv1D/Squeeze:output:0*Conv1D/BatchToSpaceND/block_shape:output:0$Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1975351

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       `
PadPadinputsPad/paddings:output:0*
T0*,
_output_shapes
:����������^
Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:}
3Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
5Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
0Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
-Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       k
!Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:w
Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
Conv1D/SpaceToBatchNDSpaceToBatchNDPad:output:0*Conv1D/SpaceToBatchND/block_shape:output:0'Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsConv1D/SpaceToBatchND:output:0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������k
!Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:t
Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
Conv1D/BatchToSpaceNDBatchToSpaceNDConv1D/Squeeze:output:0*Conv1D/BatchToSpaceND/block_shape:output:0$Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�7
 __inference__traced_save_1975890
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop.
*savev2_conv1d_6_kernel_read_readvariableop,
(savev2_conv1d_6_bias_read_readvariableop.
*savev2_conv1d_7_kernel_read_readvariableop,
(savev2_conv1d_7_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop@
<savev2_multi_head_attention_query_kernel_read_readvariableop>
:savev2_multi_head_attention_query_bias_read_readvariableop>
:savev2_multi_head_attention_key_kernel_read_readvariableop<
8savev2_multi_head_attention_key_bias_read_readvariableop@
<savev2_multi_head_attention_value_kernel_read_readvariableop>
:savev2_multi_head_attention_value_bias_read_readvariableopK
Gsavev2_multi_head_attention_attention_output_kernel_read_readvariableopI
Esavev2_multi_head_attention_attention_output_bias_read_readvariableop/
+savev2_cond_1_adam_iter_read_readvariableop	1
-savev2_cond_1_adam_beta_1_read_readvariableop1
-savev2_cond_1_adam_beta_2_read_readvariableop0
,savev2_cond_1_adam_decay_read_readvariableop8
4savev2_cond_1_adam_learning_rate_read_readvariableop1
-savev2_current_loss_scale_read_readvariableop)
%savev2_good_steps_read_readvariableop	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_true_positives_2_read_readvariableop0
,savev2_false_positives_1_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableopF
Bsavev2_cond_1_adam_layer_normalization_gamma_m_read_readvariableopE
Asavev2_cond_1_adam_layer_normalization_beta_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_kernel_m_read_readvariableop8
4savev2_cond_1_adam_conv1d_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_1_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_1_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_2_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_2_bias_m_read_readvariableopH
Dsavev2_cond_1_adam_layer_normalization_1_gamma_m_read_readvariableopG
Csavev2_cond_1_adam_layer_normalization_1_beta_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_3_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_3_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_4_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_4_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_5_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_5_bias_m_read_readvariableopH
Dsavev2_cond_1_adam_layer_normalization_2_gamma_m_read_readvariableopG
Csavev2_cond_1_adam_layer_normalization_2_beta_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_6_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_6_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_7_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_7_bias_m_read_readvariableop9
5savev2_cond_1_adam_dense_kernel_m_read_readvariableop7
3savev2_cond_1_adam_dense_bias_m_read_readvariableopN
Jsavev2_cond_1_adam_multi_head_attention_query_kernel_m_read_readvariableopL
Hsavev2_cond_1_adam_multi_head_attention_query_bias_m_read_readvariableopL
Hsavev2_cond_1_adam_multi_head_attention_key_kernel_m_read_readvariableopJ
Fsavev2_cond_1_adam_multi_head_attention_key_bias_m_read_readvariableopN
Jsavev2_cond_1_adam_multi_head_attention_value_kernel_m_read_readvariableopL
Hsavev2_cond_1_adam_multi_head_attention_value_bias_m_read_readvariableopY
Usavev2_cond_1_adam_multi_head_attention_attention_output_kernel_m_read_readvariableopW
Ssavev2_cond_1_adam_multi_head_attention_attention_output_bias_m_read_readvariableopF
Bsavev2_cond_1_adam_layer_normalization_gamma_v_read_readvariableopE
Asavev2_cond_1_adam_layer_normalization_beta_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_kernel_v_read_readvariableop8
4savev2_cond_1_adam_conv1d_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_1_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_1_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_2_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_2_bias_v_read_readvariableopH
Dsavev2_cond_1_adam_layer_normalization_1_gamma_v_read_readvariableopG
Csavev2_cond_1_adam_layer_normalization_1_beta_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_3_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_3_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_4_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_4_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_5_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_5_bias_v_read_readvariableopH
Dsavev2_cond_1_adam_layer_normalization_2_gamma_v_read_readvariableopG
Csavev2_cond_1_adam_layer_normalization_2_beta_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_6_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_6_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_7_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_7_bias_v_read_readvariableop9
5savev2_cond_1_adam_dense_kernel_v_read_readvariableop7
3savev2_cond_1_adam_dense_bias_v_read_readvariableopN
Jsavev2_cond_1_adam_multi_head_attention_query_kernel_v_read_readvariableopL
Hsavev2_cond_1_adam_multi_head_attention_query_bias_v_read_readvariableopL
Hsavev2_cond_1_adam_multi_head_attention_key_kernel_v_read_readvariableopJ
Fsavev2_cond_1_adam_multi_head_attention_key_bias_v_read_readvariableopN
Jsavev2_cond_1_adam_multi_head_attention_value_kernel_v_read_readvariableopL
Hsavev2_cond_1_adam_multi_head_attention_value_bias_v_read_readvariableopY
Usavev2_cond_1_adam_multi_head_attention_attention_output_kernel_v_read_readvariableopW
Ssavev2_cond_1_adam_multi_head_attention_attention_output_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �>
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*�=
value�=B�=tB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*�
value�B�tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �5
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop+savev2_cond_1_adam_iter_read_readvariableop-savev2_cond_1_adam_beta_1_read_readvariableop-savev2_cond_1_adam_beta_2_read_readvariableop,savev2_cond_1_adam_decay_read_readvariableop4savev2_cond_1_adam_learning_rate_read_readvariableop-savev2_current_loss_scale_read_readvariableop%savev2_good_steps_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_true_positives_2_read_readvariableop,savev2_false_positives_1_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableopBsavev2_cond_1_adam_layer_normalization_gamma_m_read_readvariableopAsavev2_cond_1_adam_layer_normalization_beta_m_read_readvariableop6savev2_cond_1_adam_conv1d_kernel_m_read_readvariableop4savev2_cond_1_adam_conv1d_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_1_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_1_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_2_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_2_bias_m_read_readvariableopDsavev2_cond_1_adam_layer_normalization_1_gamma_m_read_readvariableopCsavev2_cond_1_adam_layer_normalization_1_beta_m_read_readvariableop8savev2_cond_1_adam_conv1d_3_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_3_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_4_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_4_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_5_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_5_bias_m_read_readvariableopDsavev2_cond_1_adam_layer_normalization_2_gamma_m_read_readvariableopCsavev2_cond_1_adam_layer_normalization_2_beta_m_read_readvariableop8savev2_cond_1_adam_conv1d_6_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_6_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_7_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_7_bias_m_read_readvariableop5savev2_cond_1_adam_dense_kernel_m_read_readvariableop3savev2_cond_1_adam_dense_bias_m_read_readvariableopJsavev2_cond_1_adam_multi_head_attention_query_kernel_m_read_readvariableopHsavev2_cond_1_adam_multi_head_attention_query_bias_m_read_readvariableopHsavev2_cond_1_adam_multi_head_attention_key_kernel_m_read_readvariableopFsavev2_cond_1_adam_multi_head_attention_key_bias_m_read_readvariableopJsavev2_cond_1_adam_multi_head_attention_value_kernel_m_read_readvariableopHsavev2_cond_1_adam_multi_head_attention_value_bias_m_read_readvariableopUsavev2_cond_1_adam_multi_head_attention_attention_output_kernel_m_read_readvariableopSsavev2_cond_1_adam_multi_head_attention_attention_output_bias_m_read_readvariableopBsavev2_cond_1_adam_layer_normalization_gamma_v_read_readvariableopAsavev2_cond_1_adam_layer_normalization_beta_v_read_readvariableop6savev2_cond_1_adam_conv1d_kernel_v_read_readvariableop4savev2_cond_1_adam_conv1d_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_1_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_1_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_2_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_2_bias_v_read_readvariableopDsavev2_cond_1_adam_layer_normalization_1_gamma_v_read_readvariableopCsavev2_cond_1_adam_layer_normalization_1_beta_v_read_readvariableop8savev2_cond_1_adam_conv1d_3_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_3_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_4_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_4_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_5_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_5_bias_v_read_readvariableopDsavev2_cond_1_adam_layer_normalization_2_gamma_v_read_readvariableopCsavev2_cond_1_adam_layer_normalization_2_beta_v_read_readvariableop8savev2_cond_1_adam_conv1d_6_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_6_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_7_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_7_bias_v_read_readvariableop5savev2_cond_1_adam_dense_kernel_v_read_readvariableop3savev2_cond_1_adam_dense_bias_v_read_readvariableopJsavev2_cond_1_adam_multi_head_attention_query_kernel_v_read_readvariableopHsavev2_cond_1_adam_multi_head_attention_query_bias_v_read_readvariableopHsavev2_cond_1_adam_multi_head_attention_key_kernel_v_read_readvariableopFsavev2_cond_1_adam_multi_head_attention_key_bias_v_read_readvariableopJsavev2_cond_1_adam_multi_head_attention_value_kernel_v_read_readvariableopHsavev2_cond_1_adam_multi_head_attention_value_bias_v_read_readvariableopUsavev2_cond_1_adam_multi_head_attention_attention_output_kernel_v_read_readvariableopSsavev2_cond_1_adam_multi_head_attention_attention_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypesx
v2t		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :`:`:`@:@:@@:@:`@:@:@:@:@�:�:��:�:@�:�:�:�:��:�:��:�:	�::� : :� : :� : : �:�: : : : : : : : : : : :::::�:�:�:�:`:`:`@:@:@@:@:`@:@:@:@:@�:�:��:�:@�:�:�:�:��:�:��:�:	�::� : :� : :� : : �:�:`:`:`@:@:@@:@:`@:@:@:@:@�:�:��:�:@�:�:�:�:��:�:��:�:	�::� : :� : :� : : �:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:`: 

_output_shapes
:`:($
"
_output_shapes
:`@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:`@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:)%
#
_output_shapes
:@�:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:)%
#
_output_shapes
:@�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::)%
#
_output_shapes
:� :$ 

_output_shapes

: :)%
#
_output_shapes
:� :$ 

_output_shapes

: :)%
#
_output_shapes
:� :$ 

_output_shapes

: :)%
#
_output_shapes
: �:! 

_output_shapes	
:�:!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: : ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
::!0

_output_shapes	
:�:!1

_output_shapes	
:�:!2

_output_shapes	
:�:!3

_output_shapes	
:�: 4

_output_shapes
:`: 5

_output_shapes
:`:(6$
"
_output_shapes
:`@: 7

_output_shapes
:@:(8$
"
_output_shapes
:@@: 9

_output_shapes
:@:(:$
"
_output_shapes
:`@: ;

_output_shapes
:@: <

_output_shapes
:@: =

_output_shapes
:@:)>%
#
_output_shapes
:@�:!?

_output_shapes	
:�:*@&
$
_output_shapes
:��:!A

_output_shapes	
:�:)B%
#
_output_shapes
:@�:!C

_output_shapes	
:�:!D

_output_shapes	
:�:!E

_output_shapes	
:�:*F&
$
_output_shapes
:��:!G

_output_shapes	
:�:*H&
$
_output_shapes
:��:!I

_output_shapes	
:�:%J!

_output_shapes
:	�: K

_output_shapes
::)L%
#
_output_shapes
:� :$M 

_output_shapes

: :)N%
#
_output_shapes
:� :$O 

_output_shapes

: :)P%
#
_output_shapes
:� :$Q 

_output_shapes

: :)R%
#
_output_shapes
: �:!S

_output_shapes	
:�: T

_output_shapes
:`: U

_output_shapes
:`:(V$
"
_output_shapes
:`@: W

_output_shapes
:@:(X$
"
_output_shapes
:@@: Y

_output_shapes
:@:(Z$
"
_output_shapes
:`@: [

_output_shapes
:@: \

_output_shapes
:@: ]

_output_shapes
:@:)^%
#
_output_shapes
:@�:!_

_output_shapes	
:�:*`&
$
_output_shapes
:��:!a

_output_shapes	
:�:)b%
#
_output_shapes
:@�:!c

_output_shapes	
:�:!d

_output_shapes	
:�:!e

_output_shapes	
:�:*f&
$
_output_shapes
:��:!g

_output_shapes	
:�:*h&
$
_output_shapes
:��:!i

_output_shapes	
:�:%j!

_output_shapes
:	�: k

_output_shapes
::)l%
#
_output_shapes
:� :$m 

_output_shapes

: :)n%
#
_output_shapes
:� :$o 

_output_shapes

: :)p%
#
_output_shapes
:� :$q 

_output_shapes

: :)r%
#
_output_shapes
: �:!s

_output_shapes	
:�:t

_output_shapes
: 
�d
�
B__inference_model_layer_call_and_return_conditional_losses_1972987

inputs)
layer_normalization_1972528:`)
layer_normalization_1972530:`$
conv1d_1972552:`@
conv1d_1972554:@&
conv1d_1_1972583:@@
conv1d_1_1972585:@&
conv1d_2_1972604:`@
conv1d_2_1972606:@+
layer_normalization_1_1972665:@+
layer_normalization_1_1972667:@'
conv1d_3_1972700:@�
conv1d_3_1972702:	�(
conv1d_4_1972742:��
conv1d_4_1972744:	�'
conv1d_5_1972763:@�
conv1d_5_1972765:	�,
layer_normalization_2_1972824:	�,
layer_normalization_2_1972826:	�(
conv1d_6_1972859:��
conv1d_6_1972861:	�(
conv1d_7_1972901:��
conv1d_7_1972903:	�3
multi_head_attention_1972950:� .
multi_head_attention_1972952: 3
multi_head_attention_1972954:� .
multi_head_attention_1972956: 3
multi_head_attention_1972958:� .
multi_head_attention_1972960: 3
multi_head_attention_1972962: �+
multi_head_attention_1972964:	� 
dense_1972981:	�
dense_1972983:
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall� conv1d_3/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall� conv1d_6/StatefulPartitionedCall� conv1d_7/StatefulPartitionedCall�dense/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_1972528layer_normalization_1972530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1972527�
conv1d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0conv1d_1972552conv1d_1972554*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1972551�
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1972562�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_1_1972583conv1d_1_1972585*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1972582�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_1972604conv1d_2_1972606*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1972603�
dropout_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1972614�
tf.__operators__.add/AddV2AddV2)conv1d_2/StatefulPartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
T0*+
_output_shapes
:���������@�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_1972665layer_normalization_1_1972667*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1972664�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_3_1972700conv1d_3_1972702*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1972699�
dropout_2/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1972710�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv1d_4_1972742conv1d_4_1972744*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1972741�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_5_1972763conv1d_5_1972765*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1972762�
dropout_3/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1972773�
tf.__operators__.add_1/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0"dropout_3/PartitionedCall:output:0*
T0*,
_output_shapes
:�����������
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_1972824layer_normalization_2_1972826*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1972823�
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0conv1d_6_1972859conv1d_6_1972861*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1972858�
dropout_4/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1972869�
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0conv1d_7_1972901conv1d_7_1972903*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1972900�
dropout_5/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1972911�
tf.__operators__.add_2/AddV2AddV2 tf.__operators__.add_1/AddV2:z:0"dropout_5/PartitionedCall:output:0*
T0*,
_output_shapes
:�����������
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0multi_head_attention_1972950multi_head_attention_1972952multi_head_attention_1972954multi_head_attention_1972956multi_head_attention_1972958multi_head_attention_1972960multi_head_attention_1972962multi_head_attention_1972964*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1972949�
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:05multi_head_attention/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:�����������
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1972470�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1972981dense_1972983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1972980u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
*__inference_conv1d_5_layer_call_fn_1975150

inputs
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1972762t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1972762

inputsB
+conv1d_expanddims_1_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1972603

inputsA
+conv1d_expanddims_1_readvariableop_resource:`@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
��
�
B__inference_model_layer_call_and_return_conditional_losses_1974794

inputs?
1layer_normalization_mul_3_readvariableop_resource:`=
/layer_normalization_add_readvariableop_resource:`H
2conv1d_conv1d_expanddims_1_readvariableop_resource:`@4
&conv1d_biasadd_readvariableop_resource:@J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@6
(conv1d_1_biasadd_readvariableop_resource:@J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:`@6
(conv1d_2_biasadd_readvariableop_resource:@A
3layer_normalization_1_mul_3_readvariableop_resource:@?
1layer_normalization_1_add_readvariableop_resource:@K
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:@�7
(conv1d_3_biasadd_readvariableop_resource:	�L
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:��7
(conv1d_4_biasadd_readvariableop_resource:	�K
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:@�7
(conv1d_5_biasadd_readvariableop_resource:	�B
3layer_normalization_2_mul_3_readvariableop_resource:	�@
1layer_normalization_2_add_readvariableop_resource:	�L
4conv1d_6_conv1d_expanddims_1_readvariableop_resource:��7
(conv1d_6_biasadd_readvariableop_resource:	�L
4conv1d_7_conv1d_expanddims_1_readvariableop_resource:��7
(conv1d_7_biasadd_readvariableop_resource:	�W
@multi_head_attention_query_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_query_add_readvariableop_resource: U
>multi_head_attention_key_einsum_einsum_readvariableop_resource:� F
4multi_head_attention_key_add_readvariableop_resource: W
@multi_head_attention_value_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_value_add_readvariableop_resource: b
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: �P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:
identity��conv1d/BiasAdd/ReadVariableOp�)conv1d/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_1/BiasAdd/ReadVariableOp�+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_2/BiasAdd/ReadVariableOp�+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_3/BiasAdd/ReadVariableOp�+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_4/BiasAdd/ReadVariableOp�+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_5/BiasAdd/ReadVariableOp�+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_6/BiasAdd/ReadVariableOp�+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_7/BiasAdd/ReadVariableOp�+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�&layer_normalization/add/ReadVariableOp�(layer_normalization/mul_3/ReadVariableOp�(layer_normalization_1/add/ReadVariableOp�*layer_normalization_1/mul_3/ReadVariableOp�(layer_normalization_2/add/ReadVariableOp�*layer_normalization_2/mul_3/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOpO
layer_normalization/ShapeShapeinputs*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization/ReshapeReshapeinputs*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:���������`t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������`:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*+
_output_shapes
:���������`�
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0�
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`�
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:`*
dtype0�
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`|
conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �

conv1d/PadPadlayer_normalization/add:z:0conv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:���������`g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d/Conv1D/ExpandDims
ExpandDimsconv1d/Pad:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@b
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������@Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout/dropout/MulMulconv1d/Relu:activations:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:���������@^
dropout/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:���������@~
conv1d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_1/PadPaddropout/dropout/Mul_1:z:0conv1d_1/Pad/paddings:output:0*
T0*+
_output_shapes
:���������@i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_1/Conv1D/ExpandDims
ExpandDimsconv1d_1/Pad:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@�
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@f
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������@i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_1/dropout/MulMulconv1d_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:���������@b
dropout_1/dropout/ShapeShapeconv1d_1/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0*

seed**
seed2e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:���������@�
tf.__operators__.add/AddV2AddV2conv1d_2/BiasAdd:output:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������@i
layer_normalization_1/ShapeShapetf.__operators__.add/AddV2:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_1/ReshapeReshapetf.__operators__.add/AddV2:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:���������@�
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@�
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@~
conv1d_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_3/PadPadlayer_normalization_1/add:z:0conv1d_3/Pad/paddings:output:0*
T0*+
_output_shapes
:���������@g
conv1d_3/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
<conv1d_3/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
>conv1d_3/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
9conv1d_3/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
6conv1d_3/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        t
*conv1d_3/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
'conv1d_3/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
conv1d_3/Conv1D/SpaceToBatchNDSpaceToBatchNDconv1d_3/Pad:output:03conv1d_3/Conv1D/SpaceToBatchND/block_shape:output:00conv1d_3/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:���������@i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_3/Conv1D/ExpandDims
ExpandDims'conv1d_3/Conv1D/SpaceToBatchND:output:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������t
*conv1d_3/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:}
$conv1d_3/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
conv1d_3/Conv1D/BatchToSpaceNDBatchToSpaceND conv1d_3/Conv1D/Squeeze:output:03conv1d_3/Conv1D/BatchToSpaceND/block_shape:output:0-conv1d_3/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_3/BiasAddBiasAdd'conv1d_3/Conv1D/BatchToSpaceND:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������g
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_2/dropout/MulMulconv1d_3/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:����������b
dropout_2/dropout/ShapeShapeconv1d_3/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed**
seed2e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:�����������
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:����������~
conv1d_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_4/PadPaddropout_2/dropout/Mul_1:z:0conv1d_4/Pad/paddings:output:0*
T0*,
_output_shapes
:����������g
conv1d_4/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
<conv1d_4/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
>conv1d_4/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
9conv1d_4/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
6conv1d_4/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        t
*conv1d_4/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
'conv1d_4/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
conv1d_4/Conv1D/SpaceToBatchNDSpaceToBatchNDconv1d_4/Pad:output:03conv1d_4/Conv1D/SpaceToBatchND/block_shape:output:00conv1d_4/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������i
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_4/Conv1D/ExpandDims
ExpandDims'conv1d_4/Conv1D/SpaceToBatchND:output:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������t
*conv1d_4/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:}
$conv1d_4/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
conv1d_4/Conv1D/BatchToSpaceNDBatchToSpaceND conv1d_4/Conv1D/Squeeze:output:03conv1d_4/Conv1D/BatchToSpaceND/block_shape:output:0-conv1d_4/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_4/BiasAddBiasAdd'conv1d_4/Conv1D/BatchToSpaceND:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������g
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:����������i
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_5/Conv1D/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

����������
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_3/dropout/MulMulconv1d_4/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:����������b
dropout_3/dropout/ShapeShapeconv1d_4/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed**
seed2e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:�����������
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:�����������
tf.__operators__.add_1/AddV2AddV2conv1d_5/BiasAdd:output:0dropout_3/dropout/Mul_1:z:0*
T0*,
_output_shapes
:����������k
layer_normalization_2/ShapeShape tf.__operators__.add_1/AddV2:z:0*
T0*
_output_shapes
:s
)layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_2/strided_sliceStridedSlice$layer_normalization_2/Shape:output:02layer_normalization_2/strided_slice/stack:output:04layer_normalization_2/strided_slice/stack_1:output:04layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_2/mulMul$layer_normalization_2/mul/x:output:0,layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_2/strided_slice_1StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_1/stack:output:06layer_normalization_2/strided_slice_1/stack_1:output:06layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_2/mul_1Mullayer_normalization_2/mul:z:0.layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_2/strided_slice_2StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_2/stack:output:06layer_normalization_2/strided_slice_2/stack_1:output:06layer_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_2/mul_2Mul&layer_normalization_2/mul_2/x:output:0.layer_normalization_2/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_2/Reshape/shapePack.layer_normalization_2/Reshape/shape/0:output:0layer_normalization_2/mul_1:z:0layer_normalization_2/mul_2:z:0.layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_2/ReshapeReshape tf.__operators__.add_1/AddV2:z:0,layer_normalization_2/Reshape/shape:output:0*
T0*0
_output_shapes
:����������x
!layer_normalization_2/ones/packedPacklayer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_2/onesFill*layer_normalization_2/ones/packed:output:0)layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_2/zeros/packedPacklayer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_2/zerosFill+layer_normalization_2/zeros/packed:output:0*layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_2/FusedBatchNormV3FusedBatchNormV3&layer_normalization_2/Reshape:output:0#layer_normalization_2/ones:output:0$layer_normalization_2/zeros:output:0$layer_normalization_2/Const:output:0&layer_normalization_2/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_2/Reshape_1Reshape*layer_normalization_2/FusedBatchNormV3:y:0$layer_normalization_2/Shape:output:0*
T0*,
_output_shapes
:�����������
*layer_normalization_2/mul_3/ReadVariableOpReadVariableOp3layer_normalization_2_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_2/mul_3Mul(layer_normalization_2/Reshape_1:output:02layer_normalization_2/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
(layer_normalization_2/add/ReadVariableOpReadVariableOp1layer_normalization_2_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_2/addAddV2layer_normalization_2/mul_3:z:00layer_normalization_2/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������~
conv1d_6/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_6/PadPadlayer_normalization_2/add:z:0conv1d_6/Pad/paddings:output:0*
T0*,
_output_shapes
:����������g
conv1d_6/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
<conv1d_6/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
>conv1d_6/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
9conv1d_6/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
6conv1d_6/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       t
*conv1d_6/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
'conv1d_6/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
conv1d_6/Conv1D/SpaceToBatchNDSpaceToBatchNDconv1d_6/Pad:output:03conv1d_6/Conv1D/SpaceToBatchND/block_shape:output:00conv1d_6/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������i
conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_6/Conv1D/ExpandDims
ExpandDims'conv1d_6/Conv1D/SpaceToBatchND:output:0'conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0b
 conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_6/Conv1D/ExpandDims_1
ExpandDims3conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
conv1d_6/Conv1DConv2D#conv1d_6/Conv1D/ExpandDims:output:0%conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv1d_6/Conv1D/SqueezeSqueezeconv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������t
*conv1d_6/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:}
$conv1d_6/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
conv1d_6/Conv1D/BatchToSpaceNDBatchToSpaceND conv1d_6/Conv1D/Squeeze:output:03conv1d_6/Conv1D/BatchToSpaceND/block_shape:output:0-conv1d_6/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_6/BiasAddBiasAdd'conv1d_6/Conv1D/BatchToSpaceND:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������g
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_4/dropout/MulMulconv1d_6/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*,
_output_shapes
:����������b
dropout_4/dropout/ShapeShapeconv1d_6/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed**
seed2e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:�����������
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*,
_output_shapes
:����������~
conv1d_7/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_7/PadPaddropout_4/dropout/Mul_1:z:0conv1d_7/Pad/paddings:output:0*
T0*,
_output_shapes
:����������g
conv1d_7/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
<conv1d_7/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
>conv1d_7/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
9conv1d_7/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
6conv1d_7/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       t
*conv1d_7/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
'conv1d_7/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
conv1d_7/Conv1D/SpaceToBatchNDSpaceToBatchNDconv1d_7/Pad:output:03conv1d_7/Conv1D/SpaceToBatchND/block_shape:output:00conv1d_7/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������i
conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_7/Conv1D/ExpandDims
ExpandDims'conv1d_7/Conv1D/SpaceToBatchND:output:0'conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0b
 conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_7/Conv1D/ExpandDims_1
ExpandDims3conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
conv1d_7/Conv1DConv2D#conv1d_7/Conv1D/ExpandDims:output:0%conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv1d_7/Conv1D/SqueezeSqueezeconv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������t
*conv1d_7/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:}
$conv1d_7/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
conv1d_7/Conv1D/BatchToSpaceNDBatchToSpaceND conv1d_7/Conv1D/Squeeze:output:03conv1d_7/Conv1D/BatchToSpaceND/block_shape:output:0-conv1d_7/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_7/BiasAddBiasAdd'conv1d_7/Conv1D/BatchToSpaceND:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������g
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_5/dropout/MulMulconv1d_7/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:����������b
dropout_5/dropout/ShapeShapeconv1d_7/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed**
seed2e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:�����������
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*,
_output_shapes
:�����������
tf.__operators__.add_2/AddV2AddV2 tf.__operators__.add_1/AddV2:z:0dropout_5/dropout/Mul_1:z:0*
T0*,
_output_shapes
:�����������
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/query/einsum/EinsumEinsum tf.__operators__.add_2/AddV2:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
&multi_head_attention/key/einsum/EinsumEinsum tf.__operators__.add_2/AddV2:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/value/einsum/EinsumEinsum tf.__operators__.add_2/AddV2:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:��������� �
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/softmax/Softmax:softmax:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:��������� *
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:����������q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d/MeanMean tf.__operators__.add_3/AddV2:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp)^layer_normalization_2/add/ReadVariableOp+^layer_normalization_2/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2T
(layer_normalization_2/add/ReadVariableOp(layer_normalization_2/add/ReadVariableOp2X
*layer_normalization_2/mul_3/ReadVariableOp*layer_normalization_2/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1975141

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       `
PadPadinputsPad/paddings:output:0*
T0*,
_output_shapes
:����������^
Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:}
3Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
5Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
0Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
-Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        k
!Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:w
Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
Conv1D/SpaceToBatchNDSpaceToBatchNDPad:output:0*Conv1D/SpaceToBatchND/block_shape:output:0'Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsConv1D/SpaceToBatchND:output:0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������k
!Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:t
Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
Conv1D/BatchToSpaceNDBatchToSpaceNDConv1D/Squeeze:output:0*Conv1D/BatchToSpaceND/block_shape:output:0$Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
)__inference_dropout_layer_call_fn_1974887

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1973368s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_1974892

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_1973690
input_1
unknown:`
	unknown_0:`
	unknown_1:`@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:`@
	unknown_6:@
	unknown_7:@
	unknown_8:@ 
	unknown_9:@�

unknown_10:	�"

unknown_11:��

unknown_12:	�!

unknown_13:@�

unknown_14:	�

unknown_15:	�

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�!

unknown_21:� 

unknown_22: !

unknown_23:� 

unknown_24: !

unknown_25:� 

unknown_26: !

unknown_27: �

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1973554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������`
!
_user_specified_name	input_1
�
G
+__inference_dropout_1_layer_call_fn_1974960

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1972614d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1972614

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv1d_layer_call_and_return_conditional_losses_1974877

inputsA
+conv1d_expanddims_1_readvariableop_resource:`@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       _
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������``
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsPad:output:0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������@e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
7__inference_layer_normalization_1_layer_call_fn_1974991

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1972664s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_1972562

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_conv1d_layer_call_fn_1974859

inputs
unknown:`@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1972551s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1972858

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       `
PadPadinputsPad/paddings:output:0*
T0*,
_output_shapes
:����������^
Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:}
3Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
5Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
0Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
-Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       k
!Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:w
Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
Conv1D/SpaceToBatchNDSpaceToBatchNDPad:output:0*Conv1D/SpaceToBatchND/block_shape:output:0'Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsConv1D/SpaceToBatchND:output:0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������k
!Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:t
Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
Conv1D/BatchToSpaceNDBatchToSpaceNDConv1D/Squeeze:output:0*Conv1D/BatchToSpaceND/block_shape:output:0$Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_5_layer_call_and_return_conditional_losses_1975366

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_1973239

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_dropout_4_layer_call_fn_1975291

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1972869e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_1975180

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
7__inference_layer_normalization_2_layer_call_fn_1975201

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1972823t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1975165

inputsB
+conv1d_expanddims_1_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�)
�
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1975491	
query	
valueB
+query_einsum_einsum_readvariableop_resource:� 3
!query_add_readvariableop_resource: @
)key_einsum_einsum_readvariableop_resource:� 1
key_add_readvariableop_resource: B
+value_einsum_einsum_readvariableop_resource:� 3
!value_add_readvariableop_resource: M
6attention_output_einsum_einsum_readvariableop_resource: �;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:��������� �
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumsoftmax/Softmax:softmax:0value/add:z:0*
N*
T0*/
_output_shapes
:��������� *
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:����������

_user_specified_namequery:SO
,
_output_shapes
:����������

_user_specified_namevalue
�
�
"__inference__wrapped_model_1972460
input_1E
7model_layer_normalization_mul_3_readvariableop_resource:`C
5model_layer_normalization_add_readvariableop_resource:`N
8model_conv1d_conv1d_expanddims_1_readvariableop_resource:`@:
,model_conv1d_biasadd_readvariableop_resource:@P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@<
.model_conv1d_1_biasadd_readvariableop_resource:@P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:`@<
.model_conv1d_2_biasadd_readvariableop_resource:@G
9model_layer_normalization_1_mul_3_readvariableop_resource:@E
7model_layer_normalization_1_add_readvariableop_resource:@Q
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:@�=
.model_conv1d_3_biasadd_readvariableop_resource:	�R
:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource:��=
.model_conv1d_4_biasadd_readvariableop_resource:	�Q
:model_conv1d_5_conv1d_expanddims_1_readvariableop_resource:@�=
.model_conv1d_5_biasadd_readvariableop_resource:	�H
9model_layer_normalization_2_mul_3_readvariableop_resource:	�F
7model_layer_normalization_2_add_readvariableop_resource:	�R
:model_conv1d_6_conv1d_expanddims_1_readvariableop_resource:��=
.model_conv1d_6_biasadd_readvariableop_resource:	�R
:model_conv1d_7_conv1d_expanddims_1_readvariableop_resource:��=
.model_conv1d_7_biasadd_readvariableop_resource:	�]
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:� N
<model_multi_head_attention_query_add_readvariableop_resource: [
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:� L
:model_multi_head_attention_key_add_readvariableop_resource: ]
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:� N
<model_multi_head_attention_value_add_readvariableop_resource: h
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: �V
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:	�=
*model_dense_matmul_readvariableop_resource:	�9
+model_dense_biasadd_readvariableop_resource:
identity��#model/conv1d/BiasAdd/ReadVariableOp�/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_1/BiasAdd/ReadVariableOp�1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_2/BiasAdd/ReadVariableOp�1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_3/BiasAdd/ReadVariableOp�1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_4/BiasAdd/ReadVariableOp�1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_5/BiasAdd/ReadVariableOp�1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_6/BiasAdd/ReadVariableOp�1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp�%model/conv1d_7/BiasAdd/ReadVariableOp�1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�,model/layer_normalization/add/ReadVariableOp�.model/layer_normalization/mul_3/ReadVariableOp�.model/layer_normalization_1/add/ReadVariableOp�0model/layer_normalization_1/mul_3/ReadVariableOp�.model/layer_normalization_2/add/ReadVariableOp�0model/layer_normalization_2/mul_3/ReadVariableOp�>model/multi_head_attention/attention_output/add/ReadVariableOp�Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�1model/multi_head_attention/key/add/ReadVariableOp�;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp�3model/multi_head_attention/query/add/ReadVariableOp�=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp�3model/multi_head_attention/value/add/ReadVariableOp�=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpV
model/layer_normalization/ShapeShapeinput_1*
T0*
_output_shapes
:w
-model/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model/layer_normalization/strided_sliceStridedSlice(model/layer_normalization/Shape:output:06model/layer_normalization/strided_slice/stack:output:08model/layer_normalization/strided_slice/stack_1:output:08model/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
model/layer_normalization/mulMul(model/layer_normalization/mul/x:output:00model/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: y
/model/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)model/layer_normalization/strided_slice_1StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_1/stack:output:0:model/layer_normalization/strided_slice_1/stack_1:output:0:model/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
model/layer_normalization/mul_1Mul!model/layer_normalization/mul:z:02model/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: y
/model/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)model/layer_normalization/strided_slice_2StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_2/stack:output:0:model/layer_normalization/strided_slice_2/stack_1:output:0:model/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
model/layer_normalization/mul_2Mul*model/layer_normalization/mul_2/x:output:02model/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: k
)model/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :k
)model/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
'model/layer_normalization/Reshape/shapePack2model/layer_normalization/Reshape/shape/0:output:0#model/layer_normalization/mul_1:z:0#model/layer_normalization/mul_2:z:02model/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
!model/layer_normalization/ReshapeReshapeinput_10model/layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:���������`�
%model/layer_normalization/ones/packedPack#model/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:i
$model/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/layer_normalization/onesFill.model/layer_normalization/ones/packed:output:0-model/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:����������
&model/layer_normalization/zeros/packedPack#model/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:j
%model/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model/layer_normalization/zerosFill/model/layer_normalization/zeros/packed:output:0.model/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������b
model/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!model/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
*model/layer_normalization/FusedBatchNormV3FusedBatchNormV3*model/layer_normalization/Reshape:output:0'model/layer_normalization/ones:output:0(model/layer_normalization/zeros:output:0(model/layer_normalization/Const:output:0*model/layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������`:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
#model/layer_normalization/Reshape_1Reshape.model/layer_normalization/FusedBatchNormV3:y:0(model/layer_normalization/Shape:output:0*
T0*+
_output_shapes
:���������`�
.model/layer_normalization/mul_3/ReadVariableOpReadVariableOp7model_layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0�
model/layer_normalization/mul_3Mul,model/layer_normalization/Reshape_1:output:06model/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`�
,model/layer_normalization/add/ReadVariableOpReadVariableOp5model_layer_normalization_add_readvariableop_resource*
_output_shapes
:`*
dtype0�
model/layer_normalization/addAddV2#model/layer_normalization/mul_3:z:04model/layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`�
model/conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
model/conv1d/PadPad!model/layer_normalization/add:z:0"model/conv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:���������`m
"model/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/conv1d/Conv1D/ExpandDims
ExpandDimsmodel/conv1d/Pad:output:0+model/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0f
$model/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
 model/conv1d/Conv1D/ExpandDims_1
ExpandDims7model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
model/conv1d/Conv1DConv2D'model/conv1d/Conv1D/ExpandDims:output:0)model/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
model/conv1d/Conv1D/SqueezeSqueezemodel/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv1d/BiasAddBiasAdd$model/conv1d/Conv1D/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@n
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������@y
model/dropout/IdentityIdentitymodel/conv1d/Relu:activations:0*
T0*+
_output_shapes
:���������@�
model/conv1d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
model/conv1d_1/PadPadmodel/dropout/Identity:output:0$model/conv1d_1/Pad/paddings:output:0*
T0*+
_output_shapes
:���������@o
$model/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_1/Conv1D/ExpandDims
ExpandDimsmodel/conv1d_1/Pad:output:0-model/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0h
&model/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_1/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@�
model/conv1d_1/Conv1DConv2D)model/conv1d_1/Conv1D/ExpandDims:output:0+model/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
model/conv1d_1/Conv1D/SqueezeSqueezemodel/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/Conv1D/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@r
model/conv1d_1/ReluRelumodel/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������@o
$model/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_2/Conv1D/ExpandDims
ExpandDimsinput_1-model/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0h
&model/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_2/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
model/conv1d_2/Conv1DConv2D)model/conv1d_2/Conv1D/ExpandDims:output:0+model/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
model/conv1d_2/Conv1D/SqueezeSqueezemodel/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
%model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv1d_2/BiasAddBiasAdd&model/conv1d_2/Conv1D/Squeeze:output:0-model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@}
model/dropout_1/IdentityIdentity!model/conv1d_1/Relu:activations:0*
T0*+
_output_shapes
:���������@�
 model/tf.__operators__.add/AddV2AddV2model/conv1d_2/BiasAdd:output:0!model/dropout_1/Identity:output:0*
T0*+
_output_shapes
:���������@u
!model/layer_normalization_1/ShapeShape$model/tf.__operators__.add/AddV2:z:0*
T0*
_output_shapes
:y
/model/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)model/layer_normalization_1/strided_sliceStridedSlice*model/layer_normalization_1/Shape:output:08model/layer_normalization_1/strided_slice/stack:output:0:model/layer_normalization_1/strided_slice/stack_1:output:0:model/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
model/layer_normalization_1/mulMul*model/layer_normalization_1/mul/x:output:02model/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model/layer_normalization_1/strided_slice_1StridedSlice*model/layer_normalization_1/Shape:output:0:model/layer_normalization_1/strided_slice_1/stack:output:0<model/layer_normalization_1/strided_slice_1/stack_1:output:0<model/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
!model/layer_normalization_1/mul_1Mul#model/layer_normalization_1/mul:z:04model/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model/layer_normalization_1/strided_slice_2StridedSlice*model/layer_normalization_1/Shape:output:0:model/layer_normalization_1/strided_slice_2/stack:output:0<model/layer_normalization_1/strided_slice_2/stack_1:output:0<model/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
!model/layer_normalization_1/mul_2Mul,model/layer_normalization_1/mul_2/x:output:04model/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
)model/layer_normalization_1/Reshape/shapePack4model/layer_normalization_1/Reshape/shape/0:output:0%model/layer_normalization_1/mul_1:z:0%model/layer_normalization_1/mul_2:z:04model/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
#model/layer_normalization_1/ReshapeReshape$model/tf.__operators__.add/AddV2:z:02model/layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
'model/layer_normalization_1/ones/packedPack%model/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 model/layer_normalization_1/onesFill0model/layer_normalization_1/ones/packed:output:0/model/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:����������
(model/layer_normalization_1/zeros/packedPack%model/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
!model/layer_normalization_1/zerosFill1model/layer_normalization_1/zeros/packed:output:00model/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������d
!model/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
,model/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_1/Reshape:output:0)model/layer_normalization_1/ones:output:0*model/layer_normalization_1/zeros:output:0*model/layer_normalization_1/Const:output:0,model/layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
%model/layer_normalization_1/Reshape_1Reshape0model/layer_normalization_1/FusedBatchNormV3:y:0*model/layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:���������@�
0model/layer_normalization_1/mul_3/ReadVariableOpReadVariableOp9model_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!model/layer_normalization_1/mul_3Mul.model/layer_normalization_1/Reshape_1:output:08model/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@�
.model/layer_normalization_1/add/ReadVariableOpReadVariableOp7model_layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/layer_normalization_1/addAddV2%model/layer_normalization_1/mul_3:z:06model/layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@�
model/conv1d_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
model/conv1d_3/PadPad#model/layer_normalization_1/add:z:0$model/conv1d_3/Pad/paddings:output:0*
T0*+
_output_shapes
:���������@m
#model/conv1d_3/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
Bmodel/conv1d_3/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Dmodel/conv1d_3/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
?model/conv1d_3/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
<model/conv1d_3/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        z
0model/conv1d_3/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
-model/conv1d_3/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
$model/conv1d_3/Conv1D/SpaceToBatchNDSpaceToBatchNDmodel/conv1d_3/Pad:output:09model/conv1d_3/Conv1D/SpaceToBatchND/block_shape:output:06model/conv1d_3/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:���������@o
$model/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_3/Conv1D/ExpandDims
ExpandDims-model/conv1d_3/Conv1D/SpaceToBatchND:output:0-model/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0h
&model/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_3/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
model/conv1d_3/Conv1DConv2D)model/conv1d_3/Conv1D/ExpandDims:output:0+model/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
model/conv1d_3/Conv1D/SqueezeSqueezemodel/conv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������z
0model/conv1d_3/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
*model/conv1d_3/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
$model/conv1d_3/Conv1D/BatchToSpaceNDBatchToSpaceND&model/conv1d_3/Conv1D/Squeeze:output:09model/conv1d_3/Conv1D/BatchToSpaceND/block_shape:output:03model/conv1d_3/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
%model/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv1d_3/BiasAddBiasAdd-model/conv1d_3/Conv1D/BatchToSpaceND:output:0-model/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������s
model/conv1d_3/ReluRelumodel/conv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������~
model/dropout_2/IdentityIdentity!model/conv1d_3/Relu:activations:0*
T0*,
_output_shapes
:�����������
model/conv1d_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
model/conv1d_4/PadPad!model/dropout_2/Identity:output:0$model/conv1d_4/Pad/paddings:output:0*
T0*,
_output_shapes
:����������m
#model/conv1d_4/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
Bmodel/conv1d_4/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Dmodel/conv1d_4/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
?model/conv1d_4/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
<model/conv1d_4/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        z
0model/conv1d_4/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
-model/conv1d_4/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
$model/conv1d_4/Conv1D/SpaceToBatchNDSpaceToBatchNDmodel/conv1d_4/Pad:output:09model/conv1d_4/Conv1D/SpaceToBatchND/block_shape:output:06model/conv1d_4/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������o
$model/conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_4/Conv1D/ExpandDims
ExpandDims-model/conv1d_4/Conv1D/SpaceToBatchND:output:0-model/conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0h
&model/conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_4/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
model/conv1d_4/Conv1DConv2D)model/conv1d_4/Conv1D/ExpandDims:output:0+model/conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
model/conv1d_4/Conv1D/SqueezeSqueezemodel/conv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������z
0model/conv1d_4/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
*model/conv1d_4/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
$model/conv1d_4/Conv1D/BatchToSpaceNDBatchToSpaceND&model/conv1d_4/Conv1D/Squeeze:output:09model/conv1d_4/Conv1D/BatchToSpaceND/block_shape:output:03model/conv1d_4/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
%model/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv1d_4/BiasAddBiasAdd-model/conv1d_4/Conv1D/BatchToSpaceND:output:0-model/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������s
model/conv1d_4/ReluRelumodel/conv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:����������o
$model/conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_5/Conv1D/ExpandDims
ExpandDims$model/tf.__operators__.add/AddV2:z:0-model/conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0h
&model/conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_5/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
model/conv1d_5/Conv1DConv2D)model/conv1d_5/Conv1D/ExpandDims:output:0+model/conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
model/conv1d_5/Conv1D/SqueezeSqueezemodel/conv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

����������
%model/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv1d_5/BiasAddBiasAdd&model/conv1d_5/Conv1D/Squeeze:output:0-model/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������~
model/dropout_3/IdentityIdentity!model/conv1d_4/Relu:activations:0*
T0*,
_output_shapes
:�����������
"model/tf.__operators__.add_1/AddV2AddV2model/conv1d_5/BiasAdd:output:0!model/dropout_3/Identity:output:0*
T0*,
_output_shapes
:����������w
!model/layer_normalization_2/ShapeShape&model/tf.__operators__.add_1/AddV2:z:0*
T0*
_output_shapes
:y
/model/layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)model/layer_normalization_2/strided_sliceStridedSlice*model/layer_normalization_2/Shape:output:08model/layer_normalization_2/strided_slice/stack:output:0:model/layer_normalization_2/strided_slice/stack_1:output:0:model/layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
model/layer_normalization_2/mulMul*model/layer_normalization_2/mul/x:output:02model/layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model/layer_normalization_2/strided_slice_1StridedSlice*model/layer_normalization_2/Shape:output:0:model/layer_normalization_2/strided_slice_1/stack:output:0<model/layer_normalization_2/strided_slice_1/stack_1:output:0<model/layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
!model/layer_normalization_2/mul_1Mul#model/layer_normalization_2/mul:z:04model/layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model/layer_normalization_2/strided_slice_2StridedSlice*model/layer_normalization_2/Shape:output:0:model/layer_normalization_2/strided_slice_2/stack:output:0<model/layer_normalization_2/strided_slice_2/stack_1:output:0<model/layer_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
!model/layer_normalization_2/mul_2Mul,model/layer_normalization_2/mul_2/x:output:04model/layer_normalization_2/strided_slice_2:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
)model/layer_normalization_2/Reshape/shapePack4model/layer_normalization_2/Reshape/shape/0:output:0%model/layer_normalization_2/mul_1:z:0%model/layer_normalization_2/mul_2:z:04model/layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
#model/layer_normalization_2/ReshapeReshape&model/tf.__operators__.add_1/AddV2:z:02model/layer_normalization_2/Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
'model/layer_normalization_2/ones/packedPack%model/layer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 model/layer_normalization_2/onesFill0model/layer_normalization_2/ones/packed:output:0/model/layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:����������
(model/layer_normalization_2/zeros/packedPack%model/layer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
!model/layer_normalization_2/zerosFill1model/layer_normalization_2/zeros/packed:output:00model/layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:���������d
!model/layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
,model/layer_normalization_2/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_2/Reshape:output:0)model/layer_normalization_2/ones:output:0*model/layer_normalization_2/zeros:output:0*model/layer_normalization_2/Const:output:0,model/layer_normalization_2/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
%model/layer_normalization_2/Reshape_1Reshape0model/layer_normalization_2/FusedBatchNormV3:y:0*model/layer_normalization_2/Shape:output:0*
T0*,
_output_shapes
:�����������
0model/layer_normalization_2/mul_3/ReadVariableOpReadVariableOp9model_layer_normalization_2_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!model/layer_normalization_2/mul_3Mul.model/layer_normalization_2/Reshape_1:output:08model/layer_normalization_2/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
.model/layer_normalization_2/add/ReadVariableOpReadVariableOp7model_layer_normalization_2_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/layer_normalization_2/addAddV2%model/layer_normalization_2/mul_3:z:06model/layer_normalization_2/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
model/conv1d_6/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
model/conv1d_6/PadPad#model/layer_normalization_2/add:z:0$model/conv1d_6/Pad/paddings:output:0*
T0*,
_output_shapes
:����������m
#model/conv1d_6/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
Bmodel/conv1d_6/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Dmodel/conv1d_6/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
?model/conv1d_6/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
<model/conv1d_6/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       z
0model/conv1d_6/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
-model/conv1d_6/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
$model/conv1d_6/Conv1D/SpaceToBatchNDSpaceToBatchNDmodel/conv1d_6/Pad:output:09model/conv1d_6/Conv1D/SpaceToBatchND/block_shape:output:06model/conv1d_6/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������o
$model/conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_6/Conv1D/ExpandDims
ExpandDims-model/conv1d_6/Conv1D/SpaceToBatchND:output:0-model/conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_6_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0h
&model/conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_6/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
model/conv1d_6/Conv1DConv2D)model/conv1d_6/Conv1D/ExpandDims:output:0+model/conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
model/conv1d_6/Conv1D/SqueezeSqueezemodel/conv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������z
0model/conv1d_6/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
*model/conv1d_6/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
$model/conv1d_6/Conv1D/BatchToSpaceNDBatchToSpaceND&model/conv1d_6/Conv1D/Squeeze:output:09model/conv1d_6/Conv1D/BatchToSpaceND/block_shape:output:03model/conv1d_6/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
%model/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv1d_6/BiasAddBiasAdd-model/conv1d_6/Conv1D/BatchToSpaceND:output:0-model/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������s
model/conv1d_6/ReluRelumodel/conv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������~
model/dropout_4/IdentityIdentity!model/conv1d_6/Relu:activations:0*
T0*,
_output_shapes
:�����������
model/conv1d_7/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
model/conv1d_7/PadPad!model/dropout_4/Identity:output:0$model/conv1d_7/Pad/paddings:output:0*
T0*,
_output_shapes
:����������m
#model/conv1d_7/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
Bmodel/conv1d_7/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
Dmodel/conv1d_7/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
?model/conv1d_7/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
<model/conv1d_7/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       z
0model/conv1d_7/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
-model/conv1d_7/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
$model/conv1d_7/Conv1D/SpaceToBatchNDSpaceToBatchNDmodel/conv1d_7/Pad:output:09model/conv1d_7/Conv1D/SpaceToBatchND/block_shape:output:06model/conv1d_7/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������o
$model/conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/conv1d_7/Conv1D/ExpandDims
ExpandDims-model/conv1d_7/Conv1D/SpaceToBatchND:output:0-model/conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_7_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0h
&model/conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"model/conv1d_7/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
model/conv1d_7/Conv1DConv2D)model/conv1d_7/Conv1D/ExpandDims:output:0+model/conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
model/conv1d_7/Conv1D/SqueezeSqueezemodel/conv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������z
0model/conv1d_7/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
*model/conv1d_7/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
$model/conv1d_7/Conv1D/BatchToSpaceNDBatchToSpaceND&model/conv1d_7/Conv1D/Squeeze:output:09model/conv1d_7/Conv1D/BatchToSpaceND/block_shape:output:03model/conv1d_7/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
%model/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv1d_7/BiasAddBiasAdd-model/conv1d_7/Conv1D/BatchToSpaceND:output:0-model/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������s
model/conv1d_7/ReluRelumodel/conv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:����������~
model/dropout_5/IdentityIdentity!model/conv1d_7/Relu:activations:0*
T0*,
_output_shapes
:�����������
"model/tf.__operators__.add_2/AddV2AddV2&model/tf.__operators__.add_1/AddV2:z:0!model/dropout_5/Identity:output:0*
T0*,
_output_shapes
:�����������
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
.model/multi_head_attention/query/einsum/EinsumEinsum&model/tf.__operators__.add_2/AddV2:z:0Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
$model/multi_head_attention/query/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
,model/multi_head_attention/key/einsum/EinsumEinsum&model/tf.__operators__.add_2/AddV2:z:0Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
"model/multi_head_attention/key/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
.model/multi_head_attention/value/einsum/EinsumEinsum&model/tf.__operators__.add_2/AddV2:z:0Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
$model/multi_head_attention/value/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� e
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
model/multi_head_attention/MulMul(model/multi_head_attention/query/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:��������� �
(model/multi_head_attention/einsum/EinsumEinsum&model/multi_head_attention/key/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
+model/multi_head_attention/dropout/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
*model/multi_head_attention/einsum_1/EinsumEinsum4model/multi_head_attention/dropout/Identity:output:0(model/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:��������� *
equationacbe,aecd->abcd�
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/model/multi_head_attention/attention_output/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
"model/tf.__operators__.add_3/AddV2AddV2&model/tf.__operators__.add_2/AddV2:z:03model/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:����������w
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
#model/global_average_pooling1d/MeanMean&model/tf.__operators__.add_3/AddV2:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense/MatMulMatMul,model/global_average_pooling1d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
model/dense/SigmoidSigmoidmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������f
IdentityIdentitymodel/dense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_2/BiasAdd/ReadVariableOp2^model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_3/BiasAdd/ReadVariableOp2^model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_4/BiasAdd/ReadVariableOp2^model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_5/BiasAdd/ReadVariableOp2^model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_6/BiasAdd/ReadVariableOp2^model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_7/BiasAdd/ReadVariableOp2^model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp-^model/layer_normalization/add/ReadVariableOp/^model/layer_normalization/mul_3/ReadVariableOp/^model/layer_normalization_1/add/ReadVariableOp1^model/layer_normalization_1/mul_3/ReadVariableOp/^model/layer_normalization_2/add/ReadVariableOp1^model/layer_normalization_2/mul_3/ReadVariableOp?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_2/BiasAdd/ReadVariableOp%model/conv1d_2/BiasAdd/ReadVariableOp2f
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_3/BiasAdd/ReadVariableOp%model/conv1d_3/BiasAdd/ReadVariableOp2f
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_4/BiasAdd/ReadVariableOp%model/conv1d_4/BiasAdd/ReadVariableOp2f
1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_5/BiasAdd/ReadVariableOp%model/conv1d_5/BiasAdd/ReadVariableOp2f
1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_6/BiasAdd/ReadVariableOp%model/conv1d_6/BiasAdd/ReadVariableOp2f
1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_7/BiasAdd/ReadVariableOp%model/conv1d_7/BiasAdd/ReadVariableOp2f
1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2\
,model/layer_normalization/add/ReadVariableOp,model/layer_normalization/add/ReadVariableOp2`
.model/layer_normalization/mul_3/ReadVariableOp.model/layer_normalization/mul_3/ReadVariableOp2`
.model/layer_normalization_1/add/ReadVariableOp.model/layer_normalization_1/add/ReadVariableOp2d
0model/layer_normalization_1/mul_3/ReadVariableOp0model/layer_normalization_1/mul_3/ReadVariableOp2`
.model/layer_normalization_2/add/ReadVariableOp.model/layer_normalization_2/add/ReadVariableOp2d
0model/layer_normalization_2/mul_3/ReadVariableOp0model/layer_normalization_2/mul_3/ReadVariableOp2�
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2�
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp:T P
+
_output_shapes
:���������`
!
_user_specified_name	input_1
�

e
F__inference_dropout_5_layer_call_and_return_conditional_losses_1973163

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_5_layer_call_and_return_conditional_losses_1975378

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1975038

inputs+
mul_3_readvariableop_resource:@)
add_readvariableop_resource:@
identity��add/ReadVariableOp�mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:���������@n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:@*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������@r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�

c
D__inference_dropout_layer_call_and_return_conditional_losses_1973368

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_1972773

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_dropout_layer_call_fn_1974882

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1972562d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�m
�
B__inference_model_layer_call_and_return_conditional_losses_1973554

inputs)
layer_normalization_1973465:`)
layer_normalization_1973467:`$
conv1d_1973470:`@
conv1d_1973472:@&
conv1d_1_1973476:@@
conv1d_1_1973478:@&
conv1d_2_1973481:`@
conv1d_2_1973483:@+
layer_normalization_1_1973488:@+
layer_normalization_1_1973490:@'
conv1d_3_1973493:@�
conv1d_3_1973495:	�(
conv1d_4_1973499:��
conv1d_4_1973501:	�'
conv1d_5_1973504:@�
conv1d_5_1973506:	�,
layer_normalization_2_1973511:	�,
layer_normalization_2_1973513:	�(
conv1d_6_1973516:��
conv1d_6_1973518:	�(
conv1d_7_1973522:��
conv1d_7_1973524:	�3
multi_head_attention_1973529:� .
multi_head_attention_1973531: 3
multi_head_attention_1973533:� .
multi_head_attention_1973535: 3
multi_head_attention_1973537:� .
multi_head_attention_1973539: 3
multi_head_attention_1973541: �+
multi_head_attention_1973543:	� 
dense_1973548:	�
dense_1973550:
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall� conv1d_3/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall� conv1d_6/StatefulPartitionedCall� conv1d_7/StatefulPartitionedCall�dense/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_1973465layer_normalization_1973467*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1972527�
conv1d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0conv1d_1973470conv1d_1973472*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1972551�
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1973368�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_1_1973476conv1d_1_1973478*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1972582�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_1973481conv1d_2_1973483*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1972603�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1973325�
tf.__operators__.add/AddV2AddV2)conv1d_2/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:���������@�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_1973488layer_normalization_1_1973490*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1972664�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_3_1973493conv1d_3_1973495*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1972699�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1973282�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv1d_4_1973499conv1d_4_1973501*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1972741�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_5_1973504conv1d_5_1973506*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1972762�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1973239�
tf.__operators__.add_1/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0*dropout_3/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:�����������
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_1973511layer_normalization_2_1973513*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1972823�
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0conv1d_6_1973516conv1d_6_1973518*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1972858�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1973196�
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0conv1d_7_1973522conv1d_7_1973524*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1972900�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1973163�
tf.__operators__.add_2/AddV2AddV2 tf.__operators__.add_1/AddV2:z:0*dropout_5/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:�����������
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0multi_head_attention_1973529multi_head_attention_1973531multi_head_attention_1973533multi_head_attention_1973535multi_head_attention_1973537multi_head_attention_1973539multi_head_attention_1973541multi_head_attention_1973543*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1973124�
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:05multi_head_attention/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:�����������
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1972470�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1973548dense_1973550*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1972980u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1975091

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_conv1d_layer_call_and_return_conditional_losses_1972551

inputsA
+conv1d_expanddims_1_readvariableop_resource:`@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       _
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������``
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsPad:output:0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������@e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�$
�
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1972527

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identity��add/ReadVariableOp�mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������`:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:���������`n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������`r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
G
+__inference_dropout_3_layer_call_fn_1975170

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1972773e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�d
�
B__inference_model_layer_call_and_return_conditional_losses_1973782
input_1)
layer_normalization_1973693:`)
layer_normalization_1973695:`$
conv1d_1973698:`@
conv1d_1973700:@&
conv1d_1_1973704:@@
conv1d_1_1973706:@&
conv1d_2_1973709:`@
conv1d_2_1973711:@+
layer_normalization_1_1973716:@+
layer_normalization_1_1973718:@'
conv1d_3_1973721:@�
conv1d_3_1973723:	�(
conv1d_4_1973727:��
conv1d_4_1973729:	�'
conv1d_5_1973732:@�
conv1d_5_1973734:	�,
layer_normalization_2_1973739:	�,
layer_normalization_2_1973741:	�(
conv1d_6_1973744:��
conv1d_6_1973746:	�(
conv1d_7_1973750:��
conv1d_7_1973752:	�3
multi_head_attention_1973757:� .
multi_head_attention_1973759: 3
multi_head_attention_1973761:� .
multi_head_attention_1973763: 3
multi_head_attention_1973765:� .
multi_head_attention_1973767: 3
multi_head_attention_1973769: �+
multi_head_attention_1973771:	� 
dense_1973776:	�
dense_1973778:
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall� conv1d_3/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall� conv1d_6/StatefulPartitionedCall� conv1d_7/StatefulPartitionedCall�dense/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_1973693layer_normalization_1973695*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1972527�
conv1d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0conv1d_1973698conv1d_1973700*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1972551�
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1972562�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_1_1973704conv1d_1_1973706*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1972582�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_2_1973709conv1d_2_1973711*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1972603�
dropout_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1972614�
tf.__operators__.add/AddV2AddV2)conv1d_2/StatefulPartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
T0*+
_output_shapes
:���������@�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_1973716layer_normalization_1_1973718*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1972664�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_3_1973721conv1d_3_1973723*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1972699�
dropout_2/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1972710�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv1d_4_1973727conv1d_4_1973729*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1972741�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_5_1973732conv1d_5_1973734*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1972762�
dropout_3/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1972773�
tf.__operators__.add_1/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0"dropout_3/PartitionedCall:output:0*
T0*,
_output_shapes
:�����������
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_1973739layer_normalization_2_1973741*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1972823�
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0conv1d_6_1973744conv1d_6_1973746*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1972858�
dropout_4/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1972869�
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0conv1d_7_1973750conv1d_7_1973752*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1972900�
dropout_5/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1972911�
tf.__operators__.add_2/AddV2AddV2 tf.__operators__.add_1/AddV2:z:0"dropout_5/PartitionedCall:output:0*
T0*,
_output_shapes
:�����������
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0multi_head_attention_1973757multi_head_attention_1973759multi_head_attention_1973761multi_head_attention_1973763multi_head_attention_1973765multi_head_attention_1973767multi_head_attention_1973769multi_head_attention_1973771*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1972949�
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:05multi_head_attention/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:�����������
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1972470�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1973776dense_1973778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1972980u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:T P
+
_output_shapes
:���������`
!
_user_specified_name	input_1
�
d
+__inference_dropout_4_layer_call_fn_1975296

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1973196t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_dropout_2_layer_call_fn_1975081

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1972710e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_conv1d_6_layer_call_fn_1975257

inputs
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1972858t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_layer_normalization_layer_call_fn_1974803

inputs
unknown:`
	unknown_0:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1972527s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_1974089

inputs
unknown:`
	unknown_0:`
	unknown_1:`@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:`@
	unknown_6:@
	unknown_7:@
	unknown_8:@ 
	unknown_9:@�

unknown_10:	�"

unknown_11:��

unknown_12:	�!

unknown_13:@�

unknown_14:	�

unknown_15:	�

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�!

unknown_21:� 

unknown_22: !

unknown_23:� 

unknown_24: !

unknown_25:� 

unknown_26: !

unknown_27: �

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1973554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
G
+__inference_dropout_5_layer_call_fn_1975356

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1972911e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
+__inference_dropout_3_layer_call_fn_1975175

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1973239t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1975502

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1973325

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1972741

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       `
PadPadinputsPad/paddings:output:0*
T0*,
_output_shapes
:����������^
Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:}
3Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
5Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
0Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
-Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        k
!Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:w
Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
Conv1D/SpaceToBatchNDSpaceToBatchNDPad:output:0*Conv1D/SpaceToBatchND/block_shape:output:0'Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsConv1D/SpaceToBatchND:output:0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������k
!Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:t
Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
Conv1D/BatchToSpaceNDBatchToSpaceNDConv1D/Squeeze:output:0*Conv1D/BatchToSpaceND/block_shape:output:0$Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling1d_layer_call_fn_1975496

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1972470i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
'__inference_dense_layer_call_fn_1975511

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1972980o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1974850

inputs+
mul_3_readvariableop_resource:`)
add_readvariableop_resource:`
identity��add/ReadVariableOp�mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������`:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:���������`n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:`*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:`*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������`r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
*__inference_conv1d_3_layer_call_fn_1975047

inputs
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1972699t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
B__inference_model_layer_call_and_return_conditional_losses_1974421

inputs?
1layer_normalization_mul_3_readvariableop_resource:`=
/layer_normalization_add_readvariableop_resource:`H
2conv1d_conv1d_expanddims_1_readvariableop_resource:`@4
&conv1d_biasadd_readvariableop_resource:@J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:@@6
(conv1d_1_biasadd_readvariableop_resource:@J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:`@6
(conv1d_2_biasadd_readvariableop_resource:@A
3layer_normalization_1_mul_3_readvariableop_resource:@?
1layer_normalization_1_add_readvariableop_resource:@K
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:@�7
(conv1d_3_biasadd_readvariableop_resource:	�L
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:��7
(conv1d_4_biasadd_readvariableop_resource:	�K
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:@�7
(conv1d_5_biasadd_readvariableop_resource:	�B
3layer_normalization_2_mul_3_readvariableop_resource:	�@
1layer_normalization_2_add_readvariableop_resource:	�L
4conv1d_6_conv1d_expanddims_1_readvariableop_resource:��7
(conv1d_6_biasadd_readvariableop_resource:	�L
4conv1d_7_conv1d_expanddims_1_readvariableop_resource:��7
(conv1d_7_biasadd_readvariableop_resource:	�W
@multi_head_attention_query_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_query_add_readvariableop_resource: U
>multi_head_attention_key_einsum_einsum_readvariableop_resource:� F
4multi_head_attention_key_add_readvariableop_resource: W
@multi_head_attention_value_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_value_add_readvariableop_resource: b
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: �P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:
identity��conv1d/BiasAdd/ReadVariableOp�)conv1d/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_1/BiasAdd/ReadVariableOp�+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_2/BiasAdd/ReadVariableOp�+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_3/BiasAdd/ReadVariableOp�+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_4/BiasAdd/ReadVariableOp�+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_5/BiasAdd/ReadVariableOp�+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_6/BiasAdd/ReadVariableOp�+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp�conv1d_7/BiasAdd/ReadVariableOp�+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�&layer_normalization/add/ReadVariableOp�(layer_normalization/mul_3/ReadVariableOp�(layer_normalization_1/add/ReadVariableOp�*layer_normalization_1/mul_3/ReadVariableOp�(layer_normalization_2/add/ReadVariableOp�*layer_normalization_2/mul_3/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOpO
layer_normalization/ShapeShapeinputs*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization/ReshapeReshapeinputs*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:���������`t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:���������u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:���������\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������`:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*+
_output_shapes
:���������`�
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:`*
dtype0�
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`�
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:`*
dtype0�
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������`|
conv1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �

conv1d/PadPadlayer_normalization/add:z:0conv1d/Pad/paddings:output:0*
T0*+
_output_shapes
:���������`g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d/Conv1D/ExpandDims
ExpandDimsconv1d/Pad:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@b
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������@m
dropout/IdentityIdentityconv1d/Relu:activations:0*
T0*+
_output_shapes
:���������@~
conv1d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_1/PadPaddropout/Identity:output:0conv1d_1/Pad/paddings:output:0*
T0*+
_output_shapes
:���������@i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_1/Conv1D/ExpandDims
ExpandDimsconv1d_1/Pad:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@�
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@f
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������@i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

����������
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@q
dropout_1/IdentityIdentityconv1d_1/Relu:activations:0*
T0*+
_output_shapes
:���������@�
tf.__operators__.add/AddV2AddV2conv1d_2/BiasAdd:output:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:���������@i
layer_normalization_1/ShapeShapetf.__operators__.add/AddV2:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_1/ReshapeReshapetf.__operators__.add/AddV2:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:���������@�
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:@*
dtype0�
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@�
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@~
conv1d_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_3/PadPadlayer_normalization_1/add:z:0conv1d_3/Pad/paddings:output:0*
T0*+
_output_shapes
:���������@g
conv1d_3/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
<conv1d_3/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
>conv1d_3/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
9conv1d_3/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
6conv1d_3/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        t
*conv1d_3/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
'conv1d_3/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
conv1d_3/Conv1D/SpaceToBatchNDSpaceToBatchNDconv1d_3/Pad:output:03conv1d_3/Conv1D/SpaceToBatchND/block_shape:output:00conv1d_3/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:���������@i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_3/Conv1D/ExpandDims
ExpandDims'conv1d_3/Conv1D/SpaceToBatchND:output:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������t
*conv1d_3/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:}
$conv1d_3/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
conv1d_3/Conv1D/BatchToSpaceNDBatchToSpaceND conv1d_3/Conv1D/Squeeze:output:03conv1d_3/Conv1D/BatchToSpaceND/block_shape:output:0-conv1d_3/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_3/BiasAddBiasAdd'conv1d_3/Conv1D/BatchToSpaceND:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������g
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������r
dropout_2/IdentityIdentityconv1d_3/Relu:activations:0*
T0*,
_output_shapes
:����������~
conv1d_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_4/PadPaddropout_2/Identity:output:0conv1d_4/Pad/paddings:output:0*
T0*,
_output_shapes
:����������g
conv1d_4/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
<conv1d_4/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
>conv1d_4/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
9conv1d_4/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
6conv1d_4/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        t
*conv1d_4/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
'conv1d_4/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
conv1d_4/Conv1D/SpaceToBatchNDSpaceToBatchNDconv1d_4/Pad:output:03conv1d_4/Conv1D/SpaceToBatchND/block_shape:output:00conv1d_4/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������i
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_4/Conv1D/ExpandDims
ExpandDims'conv1d_4/Conv1D/SpaceToBatchND:output:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������t
*conv1d_4/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:}
$conv1d_4/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
conv1d_4/Conv1D/BatchToSpaceNDBatchToSpaceND conv1d_4/Conv1D/Squeeze:output:03conv1d_4/Conv1D/BatchToSpaceND/block_shape:output:0-conv1d_4/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_4/BiasAddBiasAdd'conv1d_4/Conv1D/BatchToSpaceND:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������g
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:����������i
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_5/Conv1D/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

����������
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������r
dropout_3/IdentityIdentityconv1d_4/Relu:activations:0*
T0*,
_output_shapes
:�����������
tf.__operators__.add_1/AddV2AddV2conv1d_5/BiasAdd:output:0dropout_3/Identity:output:0*
T0*,
_output_shapes
:����������k
layer_normalization_2/ShapeShape tf.__operators__.add_1/AddV2:z:0*
T0*
_output_shapes
:s
)layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_2/strided_sliceStridedSlice$layer_normalization_2/Shape:output:02layer_normalization_2/strided_slice/stack:output:04layer_normalization_2/strided_slice/stack_1:output:04layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_2/mulMul$layer_normalization_2/mul/x:output:0,layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_2/strided_slice_1StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_1/stack:output:06layer_normalization_2/strided_slice_1/stack_1:output:06layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_2/mul_1Mullayer_normalization_2/mul:z:0.layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_2/strided_slice_2StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_2/stack:output:06layer_normalization_2/strided_slice_2/stack_1:output:06layer_normalization_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_2/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_2/mul_2Mul&layer_normalization_2/mul_2/x:output:0.layer_normalization_2/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_2/Reshape/shapePack.layer_normalization_2/Reshape/shape/0:output:0layer_normalization_2/mul_1:z:0layer_normalization_2/mul_2:z:0.layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_2/ReshapeReshape tf.__operators__.add_1/AddV2:z:0,layer_normalization_2/Reshape/shape:output:0*
T0*0
_output_shapes
:����������x
!layer_normalization_2/ones/packedPacklayer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_2/onesFill*layer_normalization_2/ones/packed:output:0)layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_2/zeros/packedPacklayer_normalization_2/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_2/zerosFill+layer_normalization_2/zeros/packed:output:0*layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_2/FusedBatchNormV3FusedBatchNormV3&layer_normalization_2/Reshape:output:0#layer_normalization_2/ones:output:0$layer_normalization_2/zeros:output:0$layer_normalization_2/Const:output:0&layer_normalization_2/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_2/Reshape_1Reshape*layer_normalization_2/FusedBatchNormV3:y:0$layer_normalization_2/Shape:output:0*
T0*,
_output_shapes
:�����������
*layer_normalization_2/mul_3/ReadVariableOpReadVariableOp3layer_normalization_2_mul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_2/mul_3Mul(layer_normalization_2/Reshape_1:output:02layer_normalization_2/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
(layer_normalization_2/add/ReadVariableOpReadVariableOp1layer_normalization_2_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
layer_normalization_2/addAddV2layer_normalization_2/mul_3:z:00layer_normalization_2/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������~
conv1d_6/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_6/PadPadlayer_normalization_2/add:z:0conv1d_6/Pad/paddings:output:0*
T0*,
_output_shapes
:����������g
conv1d_6/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
<conv1d_6/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
>conv1d_6/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
9conv1d_6/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
6conv1d_6/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       t
*conv1d_6/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
'conv1d_6/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
conv1d_6/Conv1D/SpaceToBatchNDSpaceToBatchNDconv1d_6/Pad:output:03conv1d_6/Conv1D/SpaceToBatchND/block_shape:output:00conv1d_6/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������i
conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_6/Conv1D/ExpandDims
ExpandDims'conv1d_6/Conv1D/SpaceToBatchND:output:0'conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0b
 conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_6/Conv1D/ExpandDims_1
ExpandDims3conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
conv1d_6/Conv1DConv2D#conv1d_6/Conv1D/ExpandDims:output:0%conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv1d_6/Conv1D/SqueezeSqueezeconv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������t
*conv1d_6/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:}
$conv1d_6/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
conv1d_6/Conv1D/BatchToSpaceNDBatchToSpaceND conv1d_6/Conv1D/Squeeze:output:03conv1d_6/Conv1D/BatchToSpaceND/block_shape:output:0-conv1d_6/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_6/BiasAddBiasAdd'conv1d_6/Conv1D/BatchToSpaceND:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������g
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������r
dropout_4/IdentityIdentityconv1d_6/Relu:activations:0*
T0*,
_output_shapes
:����������~
conv1d_7/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       �
conv1d_7/PadPaddropout_4/Identity:output:0conv1d_7/Pad/paddings:output:0*
T0*,
_output_shapes
:����������g
conv1d_7/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:�
<conv1d_7/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
>conv1d_7/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
9conv1d_7/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
6conv1d_7/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       t
*conv1d_7/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
'conv1d_7/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
conv1d_7/Conv1D/SpaceToBatchNDSpaceToBatchNDconv1d_7/Pad:output:03conv1d_7/Conv1D/SpaceToBatchND/block_shape:output:00conv1d_7/Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������i
conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
conv1d_7/Conv1D/ExpandDims
ExpandDims'conv1d_7/Conv1D/SpaceToBatchND:output:0'conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0b
 conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
conv1d_7/Conv1D/ExpandDims_1
ExpandDims3conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
conv1d_7/Conv1DConv2D#conv1d_7/Conv1D/ExpandDims:output:0%conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv1d_7/Conv1D/SqueezeSqueezeconv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������t
*conv1d_7/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:}
$conv1d_7/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
conv1d_7/Conv1D/BatchToSpaceNDBatchToSpaceND conv1d_7/Conv1D/Squeeze:output:03conv1d_7/Conv1D/BatchToSpaceND/block_shape:output:0-conv1d_7/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:�����������
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv1d_7/BiasAddBiasAdd'conv1d_7/Conv1D/BatchToSpaceND:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������g
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:����������r
dropout_5/IdentityIdentityconv1d_7/Relu:activations:0*
T0*,
_output_shapes
:�����������
tf.__operators__.add_2/AddV2AddV2 tf.__operators__.add_1/AddV2:z:0dropout_5/Identity:output:0*
T0*,
_output_shapes
:�����������
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/query/einsum/EinsumEinsum tf.__operators__.add_2/AddV2:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
&multi_head_attention/key/einsum/EinsumEinsum tf.__operators__.add_2/AddV2:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/value/einsum/EinsumEinsum tf.__operators__.add_2/AddV2:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:��������� �
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:��������� *
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:����������q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d/MeanMean tf.__operators__.add_3/AddV2:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp)^layer_normalization_2/add/ReadVariableOp+^layer_normalization_2/mul_3/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2T
(layer_normalization_2/add/ReadVariableOp(layer_normalization_2/add/ReadVariableOp2X
*layer_normalization_2/mul_3/ReadVariableOp*layer_normalization_2/mul_3/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
*__inference_conv1d_1_layer_call_fn_1974913

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1972582s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1972664

inputs+
mul_3_readvariableop_resource:@)
add_readvariableop_resource:@
identity��add/ReadVariableOp�mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:���������@n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:@*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:���������@r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_4_layer_call_and_return_conditional_losses_1975301

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1974970

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:���������@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1973951
input_1
unknown:`
	unknown_0:`
	unknown_1:`@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:`@
	unknown_6:@
	unknown_7:@
	unknown_8:@ 
	unknown_9:@�

unknown_10:	�"

unknown_11:��

unknown_12:	�!

unknown_13:@�

unknown_14:	�

unknown_15:	�

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�!

unknown_21:� 

unknown_22: !

unknown_23:� 

unknown_24: !

unknown_25:� 

unknown_26: !

unknown_27: �

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__wrapped_model_1972460o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������`
!
_user_specified_name	input_1
�

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1974982

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1975286

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       `
PadPadinputsPad/paddings:output:0*
T0*,
_output_shapes
:����������^
Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:}
3Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
5Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
0Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
-Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       k
!Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:w
Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       �
Conv1D/SpaceToBatchNDSpaceToBatchNDPad:output:0*Conv1D/SpaceToBatchND/block_shape:output:0'Conv1D/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:����������`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsConv1D/SpaceToBatchND:output:0Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:�����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������k
!Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:t
Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       �
Conv1D/BatchToSpaceNDBatchToSpaceNDConv1D/Squeeze:output:0*Conv1D/BatchToSpaceND/block_shape:output:0$Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1972582

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       _
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������@`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsPad:output:0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������@e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
+__inference_dropout_1_layer_call_fn_1974965

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1973325s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1975076

inputsB
+conv1d_expanddims_1_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       _
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������@^
Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:}
3Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
5Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
0Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
-Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        k
!Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:w
Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
Conv1D/SpaceToBatchNDSpaceToBatchNDPad:output:0*Conv1D/SpaceToBatchND/block_shape:output:0'Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:���������@`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsConv1D/SpaceToBatchND:output:0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������k
!Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:t
Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
Conv1D/BatchToSpaceNDBatchToSpaceNDConv1D/Squeeze:output:0*Conv1D/BatchToSpaceND/block_shape:output:0$Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�P
#__inference__traced_restore_1976245
file_prefix8
*assignvariableop_layer_normalization_gamma:`9
+assignvariableop_1_layer_normalization_beta:`6
 assignvariableop_2_conv1d_kernel:`@,
assignvariableop_3_conv1d_bias:@8
"assignvariableop_4_conv1d_1_kernel:@@.
 assignvariableop_5_conv1d_1_bias:@8
"assignvariableop_6_conv1d_2_kernel:`@.
 assignvariableop_7_conv1d_2_bias:@<
.assignvariableop_8_layer_normalization_1_gamma:@;
-assignvariableop_9_layer_normalization_1_beta:@:
#assignvariableop_10_conv1d_3_kernel:@�0
!assignvariableop_11_conv1d_3_bias:	�;
#assignvariableop_12_conv1d_4_kernel:��0
!assignvariableop_13_conv1d_4_bias:	�:
#assignvariableop_14_conv1d_5_kernel:@�0
!assignvariableop_15_conv1d_5_bias:	�>
/assignvariableop_16_layer_normalization_2_gamma:	�=
.assignvariableop_17_layer_normalization_2_beta:	�;
#assignvariableop_18_conv1d_6_kernel:��0
!assignvariableop_19_conv1d_6_bias:	�;
#assignvariableop_20_conv1d_7_kernel:��0
!assignvariableop_21_conv1d_7_bias:	�3
 assignvariableop_22_dense_kernel:	�,
assignvariableop_23_dense_bias:L
5assignvariableop_24_multi_head_attention_query_kernel:� E
3assignvariableop_25_multi_head_attention_query_bias: J
3assignvariableop_26_multi_head_attention_key_kernel:� C
1assignvariableop_27_multi_head_attention_key_bias: L
5assignvariableop_28_multi_head_attention_value_kernel:� E
3assignvariableop_29_multi_head_attention_value_bias: W
@assignvariableop_30_multi_head_attention_attention_output_kernel: �M
>assignvariableop_31_multi_head_attention_attention_output_bias:	�.
$assignvariableop_32_cond_1_adam_iter:	 0
&assignvariableop_33_cond_1_adam_beta_1: 0
&assignvariableop_34_cond_1_adam_beta_2: /
%assignvariableop_35_cond_1_adam_decay: 7
-assignvariableop_36_cond_1_adam_learning_rate: 0
&assignvariableop_37_current_loss_scale: (
assignvariableop_38_good_steps:	 %
assignvariableop_39_total_1: %
assignvariableop_40_count_1: #
assignvariableop_41_total: #
assignvariableop_42_count: 2
$assignvariableop_43_true_positives_2:3
%assignvariableop_44_false_positives_1:2
$assignvariableop_45_true_positives_1:3
%assignvariableop_46_false_negatives_1:1
"assignvariableop_47_true_positives:	�1
"assignvariableop_48_true_negatives:	�2
#assignvariableop_49_false_positives:	�2
#assignvariableop_50_false_negatives:	�I
;assignvariableop_51_cond_1_adam_layer_normalization_gamma_m:`H
:assignvariableop_52_cond_1_adam_layer_normalization_beta_m:`E
/assignvariableop_53_cond_1_adam_conv1d_kernel_m:`@;
-assignvariableop_54_cond_1_adam_conv1d_bias_m:@G
1assignvariableop_55_cond_1_adam_conv1d_1_kernel_m:@@=
/assignvariableop_56_cond_1_adam_conv1d_1_bias_m:@G
1assignvariableop_57_cond_1_adam_conv1d_2_kernel_m:`@=
/assignvariableop_58_cond_1_adam_conv1d_2_bias_m:@K
=assignvariableop_59_cond_1_adam_layer_normalization_1_gamma_m:@J
<assignvariableop_60_cond_1_adam_layer_normalization_1_beta_m:@H
1assignvariableop_61_cond_1_adam_conv1d_3_kernel_m:@�>
/assignvariableop_62_cond_1_adam_conv1d_3_bias_m:	�I
1assignvariableop_63_cond_1_adam_conv1d_4_kernel_m:��>
/assignvariableop_64_cond_1_adam_conv1d_4_bias_m:	�H
1assignvariableop_65_cond_1_adam_conv1d_5_kernel_m:@�>
/assignvariableop_66_cond_1_adam_conv1d_5_bias_m:	�L
=assignvariableop_67_cond_1_adam_layer_normalization_2_gamma_m:	�K
<assignvariableop_68_cond_1_adam_layer_normalization_2_beta_m:	�I
1assignvariableop_69_cond_1_adam_conv1d_6_kernel_m:��>
/assignvariableop_70_cond_1_adam_conv1d_6_bias_m:	�I
1assignvariableop_71_cond_1_adam_conv1d_7_kernel_m:��>
/assignvariableop_72_cond_1_adam_conv1d_7_bias_m:	�A
.assignvariableop_73_cond_1_adam_dense_kernel_m:	�:
,assignvariableop_74_cond_1_adam_dense_bias_m:Z
Cassignvariableop_75_cond_1_adam_multi_head_attention_query_kernel_m:� S
Aassignvariableop_76_cond_1_adam_multi_head_attention_query_bias_m: X
Aassignvariableop_77_cond_1_adam_multi_head_attention_key_kernel_m:� Q
?assignvariableop_78_cond_1_adam_multi_head_attention_key_bias_m: Z
Cassignvariableop_79_cond_1_adam_multi_head_attention_value_kernel_m:� S
Aassignvariableop_80_cond_1_adam_multi_head_attention_value_bias_m: e
Nassignvariableop_81_cond_1_adam_multi_head_attention_attention_output_kernel_m: �[
Lassignvariableop_82_cond_1_adam_multi_head_attention_attention_output_bias_m:	�I
;assignvariableop_83_cond_1_adam_layer_normalization_gamma_v:`H
:assignvariableop_84_cond_1_adam_layer_normalization_beta_v:`E
/assignvariableop_85_cond_1_adam_conv1d_kernel_v:`@;
-assignvariableop_86_cond_1_adam_conv1d_bias_v:@G
1assignvariableop_87_cond_1_adam_conv1d_1_kernel_v:@@=
/assignvariableop_88_cond_1_adam_conv1d_1_bias_v:@G
1assignvariableop_89_cond_1_adam_conv1d_2_kernel_v:`@=
/assignvariableop_90_cond_1_adam_conv1d_2_bias_v:@K
=assignvariableop_91_cond_1_adam_layer_normalization_1_gamma_v:@J
<assignvariableop_92_cond_1_adam_layer_normalization_1_beta_v:@H
1assignvariableop_93_cond_1_adam_conv1d_3_kernel_v:@�>
/assignvariableop_94_cond_1_adam_conv1d_3_bias_v:	�I
1assignvariableop_95_cond_1_adam_conv1d_4_kernel_v:��>
/assignvariableop_96_cond_1_adam_conv1d_4_bias_v:	�H
1assignvariableop_97_cond_1_adam_conv1d_5_kernel_v:@�>
/assignvariableop_98_cond_1_adam_conv1d_5_bias_v:	�L
=assignvariableop_99_cond_1_adam_layer_normalization_2_gamma_v:	�L
=assignvariableop_100_cond_1_adam_layer_normalization_2_beta_v:	�J
2assignvariableop_101_cond_1_adam_conv1d_6_kernel_v:��?
0assignvariableop_102_cond_1_adam_conv1d_6_bias_v:	�J
2assignvariableop_103_cond_1_adam_conv1d_7_kernel_v:��?
0assignvariableop_104_cond_1_adam_conv1d_7_bias_v:	�B
/assignvariableop_105_cond_1_adam_dense_kernel_v:	�;
-assignvariableop_106_cond_1_adam_dense_bias_v:[
Dassignvariableop_107_cond_1_adam_multi_head_attention_query_kernel_v:� T
Bassignvariableop_108_cond_1_adam_multi_head_attention_query_bias_v: Y
Bassignvariableop_109_cond_1_adam_multi_head_attention_key_kernel_v:� R
@assignvariableop_110_cond_1_adam_multi_head_attention_key_bias_v: [
Dassignvariableop_111_cond_1_adam_multi_head_attention_value_kernel_v:� T
Bassignvariableop_112_cond_1_adam_multi_head_attention_value_bias_v: f
Oassignvariableop_113_cond_1_adam_multi_head_attention_attention_output_kernel_v: �\
Massignvariableop_114_cond_1_adam_multi_head_attention_attention_output_bias_v:	�
identity_116��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�>
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*�=
value�=B�=tB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*�
value�B�tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypesx
v2t		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv1d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_layer_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv1d_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv1d_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv1d_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv1d_4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv1d_5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv1d_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_layer_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_layer_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv1d_6_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv1d_6_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv1d_7_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv1d_7_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp assignvariableop_22_dense_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_dense_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp5assignvariableop_24_multi_head_attention_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp3assignvariableop_25_multi_head_attention_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp3assignvariableop_26_multi_head_attention_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp1assignvariableop_27_multi_head_attention_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_multi_head_attention_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp3assignvariableop_29_multi_head_attention_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp@assignvariableop_30_multi_head_attention_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp>assignvariableop_31_multi_head_attention_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp$assignvariableop_32_cond_1_adam_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp&assignvariableop_33_cond_1_adam_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_cond_1_adam_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp%assignvariableop_35_cond_1_adam_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp-assignvariableop_36_cond_1_adam_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp&assignvariableop_37_current_loss_scaleIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_good_stepsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp$assignvariableop_43_true_positives_2Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp%assignvariableop_44_false_positives_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp$assignvariableop_45_true_positives_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp%assignvariableop_46_false_negatives_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp"assignvariableop_47_true_positivesIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp"assignvariableop_48_true_negativesIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp#assignvariableop_49_false_positivesIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp#assignvariableop_50_false_negativesIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp;assignvariableop_51_cond_1_adam_layer_normalization_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp:assignvariableop_52_cond_1_adam_layer_normalization_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp/assignvariableop_53_cond_1_adam_conv1d_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp-assignvariableop_54_cond_1_adam_conv1d_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp1assignvariableop_55_cond_1_adam_conv1d_1_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp/assignvariableop_56_cond_1_adam_conv1d_1_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp1assignvariableop_57_cond_1_adam_conv1d_2_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp/assignvariableop_58_cond_1_adam_conv1d_2_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp=assignvariableop_59_cond_1_adam_layer_normalization_1_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp<assignvariableop_60_cond_1_adam_layer_normalization_1_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp1assignvariableop_61_cond_1_adam_conv1d_3_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp/assignvariableop_62_cond_1_adam_conv1d_3_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp1assignvariableop_63_cond_1_adam_conv1d_4_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp/assignvariableop_64_cond_1_adam_conv1d_4_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp1assignvariableop_65_cond_1_adam_conv1d_5_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp/assignvariableop_66_cond_1_adam_conv1d_5_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp=assignvariableop_67_cond_1_adam_layer_normalization_2_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp<assignvariableop_68_cond_1_adam_layer_normalization_2_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp1assignvariableop_69_cond_1_adam_conv1d_6_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp/assignvariableop_70_cond_1_adam_conv1d_6_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp1assignvariableop_71_cond_1_adam_conv1d_7_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp/assignvariableop_72_cond_1_adam_conv1d_7_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp.assignvariableop_73_cond_1_adam_dense_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp,assignvariableop_74_cond_1_adam_dense_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpCassignvariableop_75_cond_1_adam_multi_head_attention_query_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpAassignvariableop_76_cond_1_adam_multi_head_attention_query_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpAassignvariableop_77_cond_1_adam_multi_head_attention_key_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp?assignvariableop_78_cond_1_adam_multi_head_attention_key_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOpCassignvariableop_79_cond_1_adam_multi_head_attention_value_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpAassignvariableop_80_cond_1_adam_multi_head_attention_value_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOpNassignvariableop_81_cond_1_adam_multi_head_attention_attention_output_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOpLassignvariableop_82_cond_1_adam_multi_head_attention_attention_output_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp;assignvariableop_83_cond_1_adam_layer_normalization_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp:assignvariableop_84_cond_1_adam_layer_normalization_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp/assignvariableop_85_cond_1_adam_conv1d_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp-assignvariableop_86_cond_1_adam_conv1d_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp1assignvariableop_87_cond_1_adam_conv1d_1_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp/assignvariableop_88_cond_1_adam_conv1d_1_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp1assignvariableop_89_cond_1_adam_conv1d_2_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp/assignvariableop_90_cond_1_adam_conv1d_2_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp=assignvariableop_91_cond_1_adam_layer_normalization_1_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp<assignvariableop_92_cond_1_adam_layer_normalization_1_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp1assignvariableop_93_cond_1_adam_conv1d_3_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp/assignvariableop_94_cond_1_adam_conv1d_3_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp1assignvariableop_95_cond_1_adam_conv1d_4_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp/assignvariableop_96_cond_1_adam_conv1d_4_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp1assignvariableop_97_cond_1_adam_conv1d_5_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp/assignvariableop_98_cond_1_adam_conv1d_5_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp=assignvariableop_99_cond_1_adam_layer_normalization_2_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp=assignvariableop_100_cond_1_adam_layer_normalization_2_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp2assignvariableop_101_cond_1_adam_conv1d_6_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp0assignvariableop_102_cond_1_adam_conv1d_6_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp2assignvariableop_103_cond_1_adam_conv1d_7_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp0assignvariableop_104_cond_1_adam_conv1d_7_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp/assignvariableop_105_cond_1_adam_dense_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp-assignvariableop_106_cond_1_adam_dense_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOpDassignvariableop_107_cond_1_adam_multi_head_attention_query_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOpBassignvariableop_108_cond_1_adam_multi_head_attention_query_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOpBassignvariableop_109_cond_1_adam_multi_head_attention_key_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp@assignvariableop_110_cond_1_adam_multi_head_attention_key_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOpDassignvariableop_111_cond_1_adam_multi_head_attention_value_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOpBassignvariableop_112_cond_1_adam_multi_head_attention_value_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOpOassignvariableop_113_cond_1_adam_multi_head_attention_attention_output_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOpMassignvariableop_114_cond_1_adam_multi_head_attention_attention_output_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_115Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_116IdentityIdentity_115:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_116Identity_116:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1972470

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�*
�
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1972949	
query	
valueB
+query_einsum_einsum_readvariableop_resource:� 3
!query_add_readvariableop_resource: @
)key_einsum_einsum_readvariableop_resource:� 1
key_add_readvariableop_resource: B
+value_einsum_einsum_readvariableop_resource:� 3
!value_add_readvariableop_resource: M
6attention_output_einsum_einsum_readvariableop_resource: �;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:��������� �
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:��������� *
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:����������

_user_specified_namequery:SO
,
_output_shapes
:����������

_user_specified_namevalue
�*
�
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1975457	
query	
valueB
+query_einsum_einsum_readvariableop_resource:� 3
!query_add_readvariableop_resource: @
)key_einsum_einsum_readvariableop_resource:� 1
key_add_readvariableop_resource: B
+value_einsum_einsum_readvariableop_resource:� 3
!value_add_readvariableop_resource: M
6attention_output_einsum_einsum_readvariableop_resource: �;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:��������� �
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:��������� *
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:����������

_user_specified_namequery:SO
,
_output_shapes
:����������

_user_specified_namevalue
�
�
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1974955

inputsA
+conv1d_expanddims_1_readvariableop_resource:`@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������`�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`@�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
*__inference_conv1d_7_layer_call_fn_1975322

inputs
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1972900t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1972699

inputsB
+conv1d_expanddims_1_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       _
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������@^
Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:}
3Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:�
5Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
0Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
-Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        k
!Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:w
Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        �
Conv1D/SpaceToBatchNDSpaceToBatchNDPad:output:0*Conv1D/SpaceToBatchND/block_shape:output:0'Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:���������@`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsConv1D/SpaceToBatchND:output:0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������	�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������	�*
squeeze_dims

���������k
!Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:t
Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        �
Conv1D/BatchToSpaceNDBatchToSpaceNDConv1D/Squeeze:output:0*Conv1D/BatchToSpaceND/block_shape:output:0$Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
+__inference_dropout_5_layer_call_fn_1975361

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1973163t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1975103

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_1973054
input_1
unknown:`
	unknown_0:`
	unknown_1:`@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:`@
	unknown_6:@
	unknown_7:@
	unknown_8:@ 
	unknown_9:@�

unknown_10:	�"

unknown_11:��

unknown_12:	�!

unknown_13:@�

unknown_14:	�

unknown_15:	�

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�!

unknown_21:� 

unknown_22: !

unknown_23:� 

unknown_24: !

unknown_25:� 

unknown_26: !

unknown_27: �

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1972987o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������`
!
_user_specified_name	input_1
�
�
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1974931

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOpu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       _
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������@`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsPad:output:0Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@�
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������@e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�

c
D__inference_dropout_layer_call_and_return_conditional_losses_1974904

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1972710

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_conv1d_2_layer_call_fn_1974940

inputs
unknown:`@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1972603s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�

e
F__inference_dropout_4_layer_call_and_return_conditional_losses_1973196

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_conv1d_4_layer_call_fn_1975112

inputs
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1972741t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1975248

inputs,
mul_3_readvariableop_resource:	�*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:q
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*,
_output_shapes
:����������o
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0u
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0j
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:����������r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_4_layer_call_and_return_conditional_losses_1972869

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
+__inference_dropout_2_layer_call_fn_1975086

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1973282t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1972823

inputs,
mul_3_readvariableop_resource:	�*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*p
_output_shapes^
\:����������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:q
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*,
_output_shapes
:����������o
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes	
:�*
dtype0u
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0j
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:����������r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_5_layer_call_and_return_conditional_losses_1972911

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1973124	
query	
valueB
+query_einsum_einsum_readvariableop_resource:� 3
!query_add_readvariableop_resource: @
)key_einsum_einsum_readvariableop_resource:� 1
key_add_readvariableop_resource: B
+value_einsum_einsum_readvariableop_resource:� 3
!value_add_readvariableop_resource: M
6attention_output_einsum_einsum_readvariableop_resource: �;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:��������� �
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumsoftmax/Softmax:softmax:0value/add:z:0*
N*
T0*/
_output_shapes
:��������� *
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:����������

_user_specified_namequery:SO
,
_output_shapes
:����������

_user_specified_namevalue
�
�
6__inference_multi_head_attention_layer_call_fn_1975400	
query	
value
unknown:� 
	unknown_0:  
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5: �
	unknown_6:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1972949t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:����������

_user_specified_namequery:SO
,
_output_shapes
:����������

_user_specified_namevalue
�

�
B__inference_dense_layer_call_and_return_conditional_losses_1975522

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_multi_head_attention_layer_call_fn_1975422	
query	
value
unknown:� 
	unknown_0:  
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5: �
	unknown_6:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1973124t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:����������

_user_specified_namequery:SO
,
_output_shapes
:����������

_user_specified_namevalue
�

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1973282

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dense_layer_call_and_return_conditional_losses_1972980

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_1974020

inputs
unknown:`
	unknown_0:`
	unknown_1:`@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:`@
	unknown_6:@
	unknown_7:@
	unknown_8:@ 
	unknown_9:@�

unknown_10:	�"

unknown_11:��

unknown_12:	�!

unknown_13:@�

unknown_14:	�

unknown_15:	�

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�!

unknown_21:� 

unknown_22: !

unknown_23:� 

unknown_24: !

unknown_25:� 

unknown_26: !

unknown_27: �

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1972987o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������`
 
_user_specified_nameinputs
�m
�
B__inference_model_layer_call_and_return_conditional_losses_1973874
input_1)
layer_normalization_1973785:`)
layer_normalization_1973787:`$
conv1d_1973790:`@
conv1d_1973792:@&
conv1d_1_1973796:@@
conv1d_1_1973798:@&
conv1d_2_1973801:`@
conv1d_2_1973803:@+
layer_normalization_1_1973808:@+
layer_normalization_1_1973810:@'
conv1d_3_1973813:@�
conv1d_3_1973815:	�(
conv1d_4_1973819:��
conv1d_4_1973821:	�'
conv1d_5_1973824:@�
conv1d_5_1973826:	�,
layer_normalization_2_1973831:	�,
layer_normalization_2_1973833:	�(
conv1d_6_1973836:��
conv1d_6_1973838:	�(
conv1d_7_1973842:��
conv1d_7_1973844:	�3
multi_head_attention_1973849:� .
multi_head_attention_1973851: 3
multi_head_attention_1973853:� .
multi_head_attention_1973855: 3
multi_head_attention_1973857:� .
multi_head_attention_1973859: 3
multi_head_attention_1973861: �+
multi_head_attention_1973863:	� 
dense_1973868:	�
dense_1973870:
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall� conv1d_3/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall� conv1d_6/StatefulPartitionedCall� conv1d_7/StatefulPartitionedCall�dense/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_1973785layer_normalization_1973787*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1972527�
conv1d/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0conv1d_1973790conv1d_1973792*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1972551�
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1973368�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_1_1973796conv1d_1_1973798*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1972582�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_2_1973801conv1d_2_1973803*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1972603�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_1973325�
tf.__operators__.add/AddV2AddV2)conv1d_2/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:���������@�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0layer_normalization_1_1973808layer_normalization_1_1973810*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1972664�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0conv1d_3_1973813conv1d_3_1973815*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1972699�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_1973282�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv1d_4_1973819conv1d_4_1973821*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1972741�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_5_1973824conv1d_5_1973826*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1972762�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_1973239�
tf.__operators__.add_1/AddV2AddV2)conv1d_5/StatefulPartitionedCall:output:0*dropout_3/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:�����������
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_2_1973831layer_normalization_2_1973833*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1972823�
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0conv1d_6_1973836conv1d_6_1973838*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1972858�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_4_layer_call_and_return_conditional_losses_1973196�
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0conv1d_7_1973842conv1d_7_1973844*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1972900�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_1973163�
tf.__operators__.add_2/AddV2AddV2 tf.__operators__.add_1/AddV2:z:0*dropout_5/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:�����������
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0multi_head_attention_1973849multi_head_attention_1973851multi_head_attention_1973853multi_head_attention_1973855multi_head_attention_1973857multi_head_attention_1973859multi_head_attention_1973861multi_head_attention_1973863*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1973124�
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:05multi_head_attention/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:�����������
(global_average_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1972470�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1973868dense_1973870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1972980u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:���������`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:T P
+
_output_shapes
:���������`
!
_user_specified_name	input_1
�

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_1975192

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_4_layer_call_and_return_conditional_losses_1975313

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������`9
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer-19
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer_with_weights-12
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)axis
	*gamma
+beta"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_random_generator"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias
 D_jit_compiled_convolution_op"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator"
_tf_keras_layer
(
U	keras_api"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\axis
	]gamma
^beta"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias
 g_jit_compiled_convolution_op"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
n_random_generator"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias
 w_jit_compiled_convolution_op"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
*0
+1
22
33
B4
C5
K6
L7
]8
^9
e10
f11
u12
v13
~14
15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31"
trackable_list_wrapper
�
*0
+1
22
33
B4
C5
K6
L7
]8
^9
e10
f11
u12
v13
~14
15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
'__inference_model_layer_call_fn_1973054
'__inference_model_layer_call_fn_1974020
'__inference_model_layer_call_fn_1974089
'__inference_model_layer_call_fn_1973690�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
B__inference_model_layer_call_and_return_conditional_losses_1974421
B__inference_model_layer_call_and_return_conditional_losses_1974794
B__inference_model_layer_call_and_return_conditional_losses_1973782
B__inference_model_layer_call_and_return_conditional_losses_1973874�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
"__inference__wrapped_model_1972460input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
loss_scale
�base_optimizer
	�iter
�beta_1
�beta_2

�decay
�learning_rate*m�+m�2m�3m�Bm�Cm�Km�Lm�]m�^m�em�fm�um�vm�~m�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�*v�+v�2v�3v�Bv�Cv�Kv�Lv�]v�^v�ev�fv�uv�vv�~v�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
-
�serving_default"
signature_map
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
5__inference_layer_normalization_layer_call_fn_1974803�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1974850�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
':%`2layer_normalization/gamma
&:$`2layer_normalization/beta
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv1d_layer_call_fn_1974859�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv1d_layer_call_and_return_conditional_losses_1974877�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!`@2conv1d/kernel
:@2conv1d/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_dropout_layer_call_fn_1974882
)__inference_dropout_layer_call_fn_1974887�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_dropout_layer_call_and_return_conditional_losses_1974892
D__inference_dropout_layer_call_and_return_conditional_losses_1974904�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv1d_1_layer_call_fn_1974913�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1974931�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#@@2conv1d_1/kernel
:@2conv1d_1/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv1d_2_layer_call_fn_1974940�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1974955�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#`@2conv1d_2/kernel
:@2conv1d_2/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_1_layer_call_fn_1974960
+__inference_dropout_1_layer_call_fn_1974965�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_1_layer_call_and_return_conditional_losses_1974970
F__inference_dropout_1_layer_call_and_return_conditional_losses_1974982�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
"
_generic_user_object
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_layer_normalization_1_layer_call_fn_1974991�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1975038�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
):'@2layer_normalization_1/gamma
(:&@2layer_normalization_1/beta
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv1d_3_layer_call_fn_1975047�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1975076�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
&:$@�2conv1d_3/kernel
:�2conv1d_3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_2_layer_call_fn_1975081
+__inference_dropout_2_layer_call_fn_1975086�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_2_layer_call_and_return_conditional_losses_1975091
F__inference_dropout_2_layer_call_and_return_conditional_losses_1975103�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv1d_4_layer_call_fn_1975112�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1975141�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%��2conv1d_4/kernel
:�2conv1d_4/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv1d_5_layer_call_fn_1975150�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1975165�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
&:$@�2conv1d_5/kernel
:�2conv1d_5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_3_layer_call_fn_1975170
+__inference_dropout_3_layer_call_fn_1975175�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1975180
F__inference_dropout_3_layer_call_and_return_conditional_losses_1975192�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_layer_normalization_2_layer_call_fn_1975201�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1975248�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
*:(�2layer_normalization_2/gamma
):'�2layer_normalization_2/beta
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv1d_6_layer_call_fn_1975257�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1975286�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%��2conv1d_6/kernel
:�2conv1d_6/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_4_layer_call_fn_1975291
+__inference_dropout_4_layer_call_fn_1975296�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_4_layer_call_and_return_conditional_losses_1975301
F__inference_dropout_4_layer_call_and_return_conditional_losses_1975313�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv1d_7_layer_call_fn_1975322�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1975351�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%��2conv1d_7/kernel
:�2conv1d_7/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_5_layer_call_fn_1975356
+__inference_dropout_5_layer_call_fn_1975361�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_5_layer_call_and_return_conditional_losses_1975366
F__inference_dropout_5_layer_call_and_return_conditional_losses_1975378�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
"
_generic_user_object
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_multi_head_attention_layer_call_fn_1975400
6__inference_multi_head_attention_layer_call_fn_1975422�
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1975457
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1975491�
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
:__inference_global_average_pooling1d_layer_call_fn_1975496�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1975502�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_layer_call_fn_1975511�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_layer_call_and_return_conditional_losses_1975522�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�2dense/kernel
:2
dense/bias
8:6� 2!multi_head_attention/query/kernel
1:/ 2multi_head_attention/query/bias
6:4� 2multi_head_attention/key/kernel
/:- 2multi_head_attention/key/bias
8:6� 2!multi_head_attention/value/kernel
1:/ 2multi_head_attention/value/bias
C:A �2,multi_head_attention/attention_output/kernel
9:7�2*multi_head_attention/attention_output/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_model_layer_call_fn_1973054input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1974020inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1974089inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1973690input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1974421inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1974794inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1973782input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1973874input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
H
�current_loss_scale
�
good_steps"
_generic_user_object
"
_generic_user_object
:	 (2cond_1/Adam/iter
: (2cond_1/Adam/beta_1
: (2cond_1/Adam/beta_2
: (2cond_1/Adam/decay
#:! (2cond_1/Adam/learning_rate
�B�
%__inference_signature_wrapper_1973951input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_layer_normalization_layer_call_fn_1974803inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1974850inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv1d_layer_call_fn_1974859inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv1d_layer_call_and_return_conditional_losses_1974877inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dropout_layer_call_fn_1974882inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_dropout_layer_call_fn_1974887inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_layer_call_and_return_conditional_losses_1974892inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dropout_layer_call_and_return_conditional_losses_1974904inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv1d_1_layer_call_fn_1974913inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1974931inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv1d_2_layer_call_fn_1974940inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1974955inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_1_layer_call_fn_1974960inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_1_layer_call_fn_1974965inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_1_layer_call_and_return_conditional_losses_1974970inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_1_layer_call_and_return_conditional_losses_1974982inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_layer_normalization_1_layer_call_fn_1974991inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1975038inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv1d_3_layer_call_fn_1975047inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1975076inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_2_layer_call_fn_1975081inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_2_layer_call_fn_1975086inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_2_layer_call_and_return_conditional_losses_1975091inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_2_layer_call_and_return_conditional_losses_1975103inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv1d_4_layer_call_fn_1975112inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1975141inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv1d_5_layer_call_fn_1975150inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1975165inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_3_layer_call_fn_1975170inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_3_layer_call_fn_1975175inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1975180inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_3_layer_call_and_return_conditional_losses_1975192inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_layer_normalization_2_layer_call_fn_1975201inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1975248inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv1d_6_layer_call_fn_1975257inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1975286inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_4_layer_call_fn_1975291inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_4_layer_call_fn_1975296inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_4_layer_call_and_return_conditional_losses_1975301inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_4_layer_call_and_return_conditional_losses_1975313inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv1d_7_layer_call_fn_1975322inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1975351inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_5_layer_call_fn_1975356inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_5_layer_call_fn_1975361inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_5_layer_call_and_return_conditional_losses_1975366inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_5_layer_call_and_return_conditional_losses_1975378inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_multi_head_attention_layer_call_fn_1975400queryvalue"�
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_multi_head_attention_layer_call_fn_1975422queryvalue"�
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1975457queryvalue"�
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1975491queryvalue"�
���
FullArgSpecx
argsp�m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_global_average_pooling1d_layer_call_fn_1975496inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1975502inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_layer_call_fn_1975511inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_layer_call_and_return_conditional_losses_1975522inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
v
�	variables
�	keras_api
�
thresholds
�true_positives
�false_positives"
_tf_keras_metric
v
�	variables
�	keras_api
�
thresholds
�true_positives
�false_negatives"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives"
_tf_keras_metric
: 2current_loss_scale
:	 2
good_steps
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
3:1`2'cond_1/Adam/layer_normalization/gamma/m
2:0`2&cond_1/Adam/layer_normalization/beta/m
/:-`@2cond_1/Adam/conv1d/kernel/m
%:#@2cond_1/Adam/conv1d/bias/m
1:/@@2cond_1/Adam/conv1d_1/kernel/m
':%@2cond_1/Adam/conv1d_1/bias/m
1:/`@2cond_1/Adam/conv1d_2/kernel/m
':%@2cond_1/Adam/conv1d_2/bias/m
5:3@2)cond_1/Adam/layer_normalization_1/gamma/m
4:2@2(cond_1/Adam/layer_normalization_1/beta/m
2:0@�2cond_1/Adam/conv1d_3/kernel/m
(:&�2cond_1/Adam/conv1d_3/bias/m
3:1��2cond_1/Adam/conv1d_4/kernel/m
(:&�2cond_1/Adam/conv1d_4/bias/m
2:0@�2cond_1/Adam/conv1d_5/kernel/m
(:&�2cond_1/Adam/conv1d_5/bias/m
6:4�2)cond_1/Adam/layer_normalization_2/gamma/m
5:3�2(cond_1/Adam/layer_normalization_2/beta/m
3:1��2cond_1/Adam/conv1d_6/kernel/m
(:&�2cond_1/Adam/conv1d_6/bias/m
3:1��2cond_1/Adam/conv1d_7/kernel/m
(:&�2cond_1/Adam/conv1d_7/bias/m
+:)	�2cond_1/Adam/dense/kernel/m
$:"2cond_1/Adam/dense/bias/m
D:B� 2/cond_1/Adam/multi_head_attention/query/kernel/m
=:; 2-cond_1/Adam/multi_head_attention/query/bias/m
B:@� 2-cond_1/Adam/multi_head_attention/key/kernel/m
;:9 2+cond_1/Adam/multi_head_attention/key/bias/m
D:B� 2/cond_1/Adam/multi_head_attention/value/kernel/m
=:; 2-cond_1/Adam/multi_head_attention/value/bias/m
O:M �2:cond_1/Adam/multi_head_attention/attention_output/kernel/m
E:C�28cond_1/Adam/multi_head_attention/attention_output/bias/m
3:1`2'cond_1/Adam/layer_normalization/gamma/v
2:0`2&cond_1/Adam/layer_normalization/beta/v
/:-`@2cond_1/Adam/conv1d/kernel/v
%:#@2cond_1/Adam/conv1d/bias/v
1:/@@2cond_1/Adam/conv1d_1/kernel/v
':%@2cond_1/Adam/conv1d_1/bias/v
1:/`@2cond_1/Adam/conv1d_2/kernel/v
':%@2cond_1/Adam/conv1d_2/bias/v
5:3@2)cond_1/Adam/layer_normalization_1/gamma/v
4:2@2(cond_1/Adam/layer_normalization_1/beta/v
2:0@�2cond_1/Adam/conv1d_3/kernel/v
(:&�2cond_1/Adam/conv1d_3/bias/v
3:1��2cond_1/Adam/conv1d_4/kernel/v
(:&�2cond_1/Adam/conv1d_4/bias/v
2:0@�2cond_1/Adam/conv1d_5/kernel/v
(:&�2cond_1/Adam/conv1d_5/bias/v
6:4�2)cond_1/Adam/layer_normalization_2/gamma/v
5:3�2(cond_1/Adam/layer_normalization_2/beta/v
3:1��2cond_1/Adam/conv1d_6/kernel/v
(:&�2cond_1/Adam/conv1d_6/bias/v
3:1��2cond_1/Adam/conv1d_7/kernel/v
(:&�2cond_1/Adam/conv1d_7/bias/v
+:)	�2cond_1/Adam/dense/kernel/v
$:"2cond_1/Adam/dense/bias/v
D:B� 2/cond_1/Adam/multi_head_attention/query/kernel/v
=:; 2-cond_1/Adam/multi_head_attention/query/bias/v
B:@� 2-cond_1/Adam/multi_head_attention/key/kernel/v
;:9 2+cond_1/Adam/multi_head_attention/key/bias/v
D:B� 2/cond_1/Adam/multi_head_attention/value/kernel/v
=:; 2-cond_1/Adam/multi_head_attention/value/bias/v
O:M �2:cond_1/Adam/multi_head_attention/attention_output/kernel/v
E:C�28cond_1/Adam/multi_head_attention/attention_output/bias/v�
"__inference__wrapped_model_1972460�0*+23BCKL]^efuv~����������������4�1
*�'
%�"
input_1���������`
� "-�*
(
dense�
dense����������
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1974931dBC3�0
)�&
$�!
inputs���������@
� ")�&
�
0���������@
� �
*__inference_conv1d_1_layer_call_fn_1974913WBC3�0
)�&
$�!
inputs���������@
� "����������@�
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1974955dKL3�0
)�&
$�!
inputs���������`
� ")�&
�
0���������@
� �
*__inference_conv1d_2_layer_call_fn_1974940WKL3�0
)�&
$�!
inputs���������`
� "����������@�
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1975076eef3�0
)�&
$�!
inputs���������@
� "*�'
 �
0����������
� �
*__inference_conv1d_3_layer_call_fn_1975047Xef3�0
)�&
$�!
inputs���������@
� "������������
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1975141fuv4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
*__inference_conv1d_4_layer_call_fn_1975112Yuv4�1
*�'
%�"
inputs����������
� "������������
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1975165e~3�0
)�&
$�!
inputs���������@
� "*�'
 �
0����������
� �
*__inference_conv1d_5_layer_call_fn_1975150X~3�0
)�&
$�!
inputs���������@
� "������������
E__inference_conv1d_6_layer_call_and_return_conditional_losses_1975286h��4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
*__inference_conv1d_6_layer_call_fn_1975257[��4�1
*�'
%�"
inputs����������
� "������������
E__inference_conv1d_7_layer_call_and_return_conditional_losses_1975351h��4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
*__inference_conv1d_7_layer_call_fn_1975322[��4�1
*�'
%�"
inputs����������
� "������������
C__inference_conv1d_layer_call_and_return_conditional_losses_1974877d233�0
)�&
$�!
inputs���������`
� ")�&
�
0���������@
� �
(__inference_conv1d_layer_call_fn_1974859W233�0
)�&
$�!
inputs���������`
� "����������@�
B__inference_dense_layer_call_and_return_conditional_losses_1975522_��0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
'__inference_dense_layer_call_fn_1975511R��0�-
&�#
!�
inputs����������
� "�����������
F__inference_dropout_1_layer_call_and_return_conditional_losses_1974970d7�4
-�*
$�!
inputs���������@
p 
� ")�&
�
0���������@
� �
F__inference_dropout_1_layer_call_and_return_conditional_losses_1974982d7�4
-�*
$�!
inputs���������@
p
� ")�&
�
0���������@
� �
+__inference_dropout_1_layer_call_fn_1974960W7�4
-�*
$�!
inputs���������@
p 
� "����������@�
+__inference_dropout_1_layer_call_fn_1974965W7�4
-�*
$�!
inputs���������@
p
� "����������@�
F__inference_dropout_2_layer_call_and_return_conditional_losses_1975091f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
F__inference_dropout_2_layer_call_and_return_conditional_losses_1975103f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
+__inference_dropout_2_layer_call_fn_1975081Y8�5
.�+
%�"
inputs����������
p 
� "������������
+__inference_dropout_2_layer_call_fn_1975086Y8�5
.�+
%�"
inputs����������
p
� "������������
F__inference_dropout_3_layer_call_and_return_conditional_losses_1975180f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
F__inference_dropout_3_layer_call_and_return_conditional_losses_1975192f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
+__inference_dropout_3_layer_call_fn_1975170Y8�5
.�+
%�"
inputs����������
p 
� "������������
+__inference_dropout_3_layer_call_fn_1975175Y8�5
.�+
%�"
inputs����������
p
� "������������
F__inference_dropout_4_layer_call_and_return_conditional_losses_1975301f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
F__inference_dropout_4_layer_call_and_return_conditional_losses_1975313f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
+__inference_dropout_4_layer_call_fn_1975291Y8�5
.�+
%�"
inputs����������
p 
� "������������
+__inference_dropout_4_layer_call_fn_1975296Y8�5
.�+
%�"
inputs����������
p
� "������������
F__inference_dropout_5_layer_call_and_return_conditional_losses_1975366f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
F__inference_dropout_5_layer_call_and_return_conditional_losses_1975378f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
+__inference_dropout_5_layer_call_fn_1975356Y8�5
.�+
%�"
inputs����������
p 
� "������������
+__inference_dropout_5_layer_call_fn_1975361Y8�5
.�+
%�"
inputs����������
p
� "������������
D__inference_dropout_layer_call_and_return_conditional_losses_1974892d7�4
-�*
$�!
inputs���������@
p 
� ")�&
�
0���������@
� �
D__inference_dropout_layer_call_and_return_conditional_losses_1974904d7�4
-�*
$�!
inputs���������@
p
� ")�&
�
0���������@
� �
)__inference_dropout_layer_call_fn_1974882W7�4
-�*
$�!
inputs���������@
p 
� "����������@�
)__inference_dropout_layer_call_fn_1974887W7�4
-�*
$�!
inputs���������@
p
� "����������@�
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_1975502{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
:__inference_global_average_pooling1d_layer_call_fn_1975496nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_1975038d]^3�0
)�&
$�!
inputs���������@
� ")�&
�
0���������@
� �
7__inference_layer_normalization_1_layer_call_fn_1974991W]^3�0
)�&
$�!
inputs���������@
� "����������@�
R__inference_layer_normalization_2_layer_call_and_return_conditional_losses_1975248h��4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
7__inference_layer_normalization_2_layer_call_fn_1975201[��4�1
*�'
%�"
inputs����������
� "������������
P__inference_layer_normalization_layer_call_and_return_conditional_losses_1974850d*+3�0
)�&
$�!
inputs���������`
� ")�&
�
0���������`
� �
5__inference_layer_normalization_layer_call_fn_1974803W*+3�0
)�&
$�!
inputs���������`
� "����������`�
B__inference_model_layer_call_and_return_conditional_losses_1973782�0*+23BCKL]^efuv~����������������<�9
2�/
%�"
input_1���������`
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1973874�0*+23BCKL]^efuv~����������������<�9
2�/
%�"
input_1���������`
p

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1974421�0*+23BCKL]^efuv~����������������;�8
1�.
$�!
inputs���������`
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1974794�0*+23BCKL]^efuv~����������������;�8
1�.
$�!
inputs���������`
p

 
� "%�"
�
0���������
� �
'__inference_model_layer_call_fn_1973054�0*+23BCKL]^efuv~����������������<�9
2�/
%�"
input_1���������`
p 

 
� "�����������
'__inference_model_layer_call_fn_1973690�0*+23BCKL]^efuv~����������������<�9
2�/
%�"
input_1���������`
p

 
� "�����������
'__inference_model_layer_call_fn_1974020�0*+23BCKL]^efuv~����������������;�8
1�.
$�!
inputs���������`
p 

 
� "�����������
'__inference_model_layer_call_fn_1974089�0*+23BCKL]^efuv~����������������;�8
1�.
$�!
inputs���������`
p

 
� "�����������
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1975457���������m�j
c�`
$�!
query����������
$�!
value����������

 

 
p 
p 
p 
� "*�'
 �
0����������
� �
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_1975491���������m�j
c�`
$�!
query����������
$�!
value����������

 

 
p 
p
p 
� "*�'
 �
0����������
� �
6__inference_multi_head_attention_layer_call_fn_1975400���������m�j
c�`
$�!
query����������
$�!
value����������

 

 
p 
p 
p 
� "������������
6__inference_multi_head_attention_layer_call_fn_1975422���������m�j
c�`
$�!
query����������
$�!
value����������

 

 
p 
p
p 
� "������������
%__inference_signature_wrapper_1973951�0*+23BCKL]^efuv~����������������?�<
� 
5�2
0
input_1%�"
input_1���������`"-�*
(
dense�
dense���������