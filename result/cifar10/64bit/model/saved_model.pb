??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
/
Sign
x"T
y"T"
Ttype:

2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??
?
tbh/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_nametbh/dense_6/kernel
y
&tbh/dense_6/kernel/Read/ReadVariableOpReadVariableOptbh/dense_6/kernel*
_output_shapes

:@*
dtype0
x
tbh/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametbh/dense_6/bias
q
$tbh/dense_6/bias/Read/ReadVariableOpReadVariableOptbh/dense_6/bias*
_output_shapes
:*
dtype0
?
tbh/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_nametbh/dense_7/kernel
z
&tbh/dense_7/kernel/Read/ReadVariableOpReadVariableOptbh/dense_7/kernel*
_output_shapes
:	?*
dtype0
x
tbh/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametbh/dense_7/bias
q
$tbh/dense_7/bias/Read/ReadVariableOpReadVariableOptbh/dense_7/bias*
_output_shapes
:*
dtype0
?
tbh/vae_encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
? ?*-
shared_nametbh/vae_encoder/dense/kernel
?
0tbh/vae_encoder/dense/kernel/Read/ReadVariableOpReadVariableOptbh/vae_encoder/dense/kernel* 
_output_shapes
:
? ?*
dtype0
?
tbh/vae_encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nametbh/vae_encoder/dense/bias
?
.tbh/vae_encoder/dense/bias/Read/ReadVariableOpReadVariableOptbh/vae_encoder/dense/bias*
_output_shapes	
:?*
dtype0
?
tbh/vae_encoder/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name tbh/vae_encoder/dense_1/kernel
?
2tbh/vae_encoder/dense_1/kernel/Read/ReadVariableOpReadVariableOptbh/vae_encoder/dense_1/kernel* 
_output_shapes
:
??*
dtype0
?
tbh/vae_encoder/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nametbh/vae_encoder/dense_1/bias
?
0tbh/vae_encoder/dense_1/bias/Read/ReadVariableOpReadVariableOptbh/vae_encoder/dense_1/bias*
_output_shapes	
:?*
dtype0
?
tbh/vae_encoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name tbh/vae_encoder/dense_2/kernel
?
2tbh/vae_encoder/dense_2/kernel/Read/ReadVariableOpReadVariableOptbh/vae_encoder/dense_2/kernel* 
_output_shapes
:
??*
dtype0
?
tbh/vae_encoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nametbh/vae_encoder/dense_2/bias
?
0tbh/vae_encoder/dense_2/bias/Read/ReadVariableOpReadVariableOptbh/vae_encoder/dense_2/bias*
_output_shapes	
:?*
dtype0
?
tbh/decoder/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nametbh/decoder/dense_3/kernel
?
.tbh/decoder/dense_3/kernel/Read/ReadVariableOpReadVariableOptbh/decoder/dense_3/kernel* 
_output_shapes
:
??*
dtype0
?
tbh/decoder/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nametbh/decoder/dense_3/bias
?
,tbh/decoder/dense_3/bias/Read/ReadVariableOpReadVariableOptbh/decoder/dense_3/bias*
_output_shapes	
:?*
dtype0
?
tbh/decoder/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *+
shared_nametbh/decoder/dense_4/kernel
?
.tbh/decoder/dense_4/kernel/Read/ReadVariableOpReadVariableOptbh/decoder/dense_4/kernel* 
_output_shapes
:
?? *
dtype0
?
tbh/decoder/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *)
shared_nametbh/decoder/dense_4/bias
?
,tbh/decoder/dense_4/bias/Read/ReadVariableOpReadVariableOptbh/decoder/dense_4/bias*
_output_shapes	
:? *
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
??*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?/
value?/B?/ B?/
?
encoder
decoder
tbn
	dis_1
	dis_2
	variables
trainable_variables
regularization_losses
		keras_api


signatures
t
fc_1

fc_2_1

fc_2_2
	variables
trainable_variables
regularization_losses
	keras_api
f
fc_1
fc_2
	variables
trainable_variables
regularization_losses
	keras_api
[
gcn
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
v
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
12
13
#14
$15
v
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
12
13
#14
$15
 
?
	variables
trainable_variables
5layer_metrics
regularization_losses

6layers
7layer_regularization_losses
8metrics
9non_trainable_variables
 
h

)kernel
*bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
h

+kernel
,bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

-kernel
.bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
*
)0
*1
+2
,3
-4
.5
*
)0
*1
+2
,3
-4
.5
 
?
	variables
trainable_variables
Flayer_metrics
regularization_losses

Glayers
Hlayer_regularization_losses
Imetrics
Jnon_trainable_variables
h

/kernel
0bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
h

1kernel
2bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api

/0
01
12
23

/0
01
12
23
 
?
	variables
trainable_variables
Slayer_metrics
regularization_losses

Tlayers
Ulayer_regularization_losses
Vmetrics
Wnon_trainable_variables
b
Xfc
Yrs
Z	variables
[trainable_variables
\regularization_losses
]	keras_api

30
41

30
41
 
?
	variables
trainable_variables
^layer_metrics
regularization_losses

_layers
`layer_regularization_losses
ametrics
bnon_trainable_variables
OM
VARIABLE_VALUEtbh/dense_6/kernel'dis_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtbh/dense_6/bias%dis_1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
 trainable_variables
clayer_metrics
!regularization_losses

dlayers
elayer_regularization_losses
fmetrics
gnon_trainable_variables
OM
VARIABLE_VALUEtbh/dense_7/kernel'dis_2/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtbh/dense_7/bias%dis_2/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
%	variables
&trainable_variables
hlayer_metrics
'regularization_losses

ilayers
jlayer_regularization_losses
kmetrics
lnon_trainable_variables
XV
VARIABLE_VALUEtbh/vae_encoder/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtbh/vae_encoder/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtbh/vae_encoder/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtbh/vae_encoder/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtbh/vae_encoder/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtbh/vae_encoder/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtbh/decoder/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtbh/decoder/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtbh/decoder/dense_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtbh/decoder/dense_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 
 
 

)0
*1

)0
*1
 
?
:	variables
;trainable_variables
mlayer_metrics
<regularization_losses

nlayers
olayer_regularization_losses
pmetrics
qnon_trainable_variables

+0
,1

+0
,1
 
?
>	variables
?trainable_variables
rlayer_metrics
@regularization_losses

slayers
tlayer_regularization_losses
umetrics
vnon_trainable_variables

-0
.1

-0
.1
 
?
B	variables
Ctrainable_variables
wlayer_metrics
Dregularization_losses

xlayers
ylayer_regularization_losses
zmetrics
{non_trainable_variables
 

0
1
2
 
 
 

/0
01

/0
01
 
?
K	variables
Ltrainable_variables
|layer_metrics
Mregularization_losses

}layers
~layer_regularization_losses
metrics
?non_trainable_variables

10
21

10
21
 
?
O	variables
Ptrainable_variables
?layer_metrics
Qregularization_losses
?layers
 ?layer_regularization_losses
?metrics
?non_trainable_variables
 

0
1
 
 
 
l

3kernel
4bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

?	keras_api

30
41

30
41
 
?
Z	variables
[trainable_variables
?layer_metrics
\regularization_losses
?layers
 ?layer_regularization_losses
?metrics
?non_trainable_variables
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

30
41

30
41
 
?
?	variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
 ?layer_regularization_losses
?metrics
?non_trainable_variables
 
 

X0
Y1
 
 
 
 
 
 
 
 
t
serving_default_input_1_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_input_1_2Placeholder*(
_output_shapes
:?????????? *
dtype0*
shape:?????????? 
|
serving_default_input_1_3Placeholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

z
serving_default_input_2Placeholder*'
_output_shapes
:?????????@*
dtype0*
shape:?????????@
|
serving_default_input_3Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1_1serving_default_input_1_2serving_default_input_1_3serving_default_input_2serving_default_input_3tbh/vae_encoder/dense/kerneltbh/vae_encoder/dense/biastbh/vae_encoder/dense_1/kerneltbh/vae_encoder/dense_1/biastbh/vae_encoder/dense_2/kerneltbh/vae_encoder/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_197195847
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&tbh/dense_6/kernel/Read/ReadVariableOp$tbh/dense_6/bias/Read/ReadVariableOp&tbh/dense_7/kernel/Read/ReadVariableOp$tbh/dense_7/bias/Read/ReadVariableOp0tbh/vae_encoder/dense/kernel/Read/ReadVariableOp.tbh/vae_encoder/dense/bias/Read/ReadVariableOp2tbh/vae_encoder/dense_1/kernel/Read/ReadVariableOp0tbh/vae_encoder/dense_1/bias/Read/ReadVariableOp2tbh/vae_encoder/dense_2/kernel/Read/ReadVariableOp0tbh/vae_encoder/dense_2/bias/Read/ReadVariableOp.tbh/decoder/dense_3/kernel/Read/ReadVariableOp,tbh/decoder/dense_3/bias/Read/ReadVariableOp.tbh/decoder/dense_4/kernel/Read/ReadVariableOp,tbh/decoder/dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_save_197196879
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametbh/dense_6/kerneltbh/dense_6/biastbh/dense_7/kerneltbh/dense_7/biastbh/vae_encoder/dense/kerneltbh/vae_encoder/dense/biastbh/vae_encoder/dense_1/kerneltbh/vae_encoder/dense_1/biastbh/vae_encoder/dense_2/kerneltbh/vae_encoder/dense_2/biastbh/decoder/dense_3/kerneltbh/decoder/dense_3/biastbh/decoder/dense_4/kerneltbh/decoder/dense_4/biasdense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference__traced_restore_197196937??
?
?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_197195432
bbn
cbn
gcn_layer_197195425
gcn_layer_197195427
identity??!gcn_layer/StatefulPartitionedCall?
PartitionedCallPartitionedCallbbn*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_build_adjacency_hamming_142320222
PartitionedCall?
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallcbnPartitionedCall:output:0gcn_layer_197195425gcn_layer_197195427*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_spectrum_conv_142320732#
!gcn_layer/StatefulPartitionedCalls
SigmoidSigmoid*gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2	
Sigmoid{
IdentityIdentitySigmoid:y:0"^gcn_layer/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:@:??????????::2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:MI
(
_output_shapes
:??????????

_user_specified_namecbn
?
?
+__inference_dense_7_layer_call_fn_197196804

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_7_layer_call_and_return_conditional_losses_1971955092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	?::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?	
?
/__inference_vae_encoder_layer_call_fn_197196599

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:@:??????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_vae_encoder_layer_call_and_return_conditional_losses_1971952912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:?????????? ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?c
?
$__inference__wrapped_model_197195199
	input_1_1
	input_1_2
	input_1_3
input_2
input_38
4tbh_vae_encoder_dense_matmul_readvariableop_resource9
5tbh_vae_encoder_dense_biasadd_readvariableop_resource:
6tbh_vae_encoder_dense_1_matmul_readvariableop_resource;
7tbh_vae_encoder_dense_1_biasadd_readvariableop_resource:
6tbh_vae_encoder_dense_2_matmul_readvariableop_resource;
7tbh_vae_encoder_dense_2_biasadd_readvariableop_resource
identity??,tbh/vae_encoder/dense/BiasAdd/ReadVariableOp?.tbh/vae_encoder/dense/BiasAdd_1/ReadVariableOp?+tbh/vae_encoder/dense/MatMul/ReadVariableOp?-tbh/vae_encoder/dense/MatMul_1/ReadVariableOp?.tbh/vae_encoder/dense_1/BiasAdd/ReadVariableOp?-tbh/vae_encoder/dense_1/MatMul/ReadVariableOp?.tbh/vae_encoder/dense_2/BiasAdd/ReadVariableOp?-tbh/vae_encoder/dense_2/MatMul/ReadVariableOpg
tbh/vae_encoder/ShapeShape	input_1_2*
T0*
_output_shapes
:2
tbh/vae_encoder/Shape?
#tbh/vae_encoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#tbh/vae_encoder/strided_slice/stack?
%tbh/vae_encoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%tbh/vae_encoder/strided_slice/stack_1?
%tbh/vae_encoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%tbh/vae_encoder/strided_slice/stack_2?
tbh/vae_encoder/strided_sliceStridedSlicetbh/vae_encoder/Shape:output:0,tbh/vae_encoder/strided_slice/stack:output:0.tbh/vae_encoder/strided_slice/stack_1:output:0.tbh/vae_encoder/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tbh/vae_encoder/strided_slice?
+tbh/vae_encoder/dense/MatMul/ReadVariableOpReadVariableOp4tbh_vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02-
+tbh/vae_encoder/dense/MatMul/ReadVariableOp?
tbh/vae_encoder/dense/MatMulMatMul	input_1_23tbh/vae_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
tbh/vae_encoder/dense/MatMul?
,tbh/vae_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp5tbh_vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,tbh/vae_encoder/dense/BiasAdd/ReadVariableOp?
tbh/vae_encoder/dense/BiasAddBiasAdd&tbh/vae_encoder/dense/MatMul:product:04tbh/vae_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
tbh/vae_encoder/dense/BiasAdd?
tbh/vae_encoder/dense/ReluRelu&tbh/vae_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tbh/vae_encoder/dense/Relu?
-tbh/vae_encoder/dense/MatMul_1/ReadVariableOpReadVariableOp4tbh_vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02/
-tbh/vae_encoder/dense/MatMul_1/ReadVariableOp?
tbh/vae_encoder/dense/MatMul_1MatMul	input_1_25tbh/vae_encoder/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
tbh/vae_encoder/dense/MatMul_1?
.tbh/vae_encoder/dense/BiasAdd_1/ReadVariableOpReadVariableOp5tbh_vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.tbh/vae_encoder/dense/BiasAdd_1/ReadVariableOp?
tbh/vae_encoder/dense/BiasAdd_1BiasAdd(tbh/vae_encoder/dense/MatMul_1:product:06tbh/vae_encoder/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
tbh/vae_encoder/dense/BiasAdd_1?
tbh/vae_encoder/dense/Relu_1Relu(tbh/vae_encoder/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
tbh/vae_encoder/dense/Relu_1?
-tbh/vae_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp6tbh_vae_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-tbh/vae_encoder/dense_1/MatMul/ReadVariableOp?
tbh/vae_encoder/dense_1/MatMulMatMul*tbh/vae_encoder/dense/Relu_1:activations:05tbh/vae_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
tbh/vae_encoder/dense_1/MatMul?
.tbh/vae_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp7tbh_vae_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.tbh/vae_encoder/dense_1/BiasAdd/ReadVariableOp?
tbh/vae_encoder/dense_1/BiasAddBiasAdd(tbh/vae_encoder/dense_1/MatMul:product:06tbh/vae_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
tbh/vae_encoder/dense_1/BiasAddp
tbh/vae_encoder/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
tbh/vae_encoder/Const?
tbh/vae_encoder/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tbh/vae_encoder/split/split_dim?
tbh/vae_encoder/splitSplit(tbh/vae_encoder/split/split_dim:output:0(tbh/vae_encoder/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
tbh/vae_encoder/split?
tbh/vae_encoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
tbh/vae_encoder/Reshape/shape?
tbh/vae_encoder/ReshapeReshapetbh/vae_encoder/split:output:0&tbh/vae_encoder/Reshape/shape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Reshape?
tbh/vae_encoder/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2!
tbh/vae_encoder/Reshape_1/shape?
tbh/vae_encoder/Reshape_1Reshapetbh/vae_encoder/split:output:1(tbh/vae_encoder/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Reshape_1?
tbh/vae_encoder/zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
tbh/vae_encoder/zeros|
tbh/vae_encoder/NegNeg tbh/vae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Negs
tbh/vae_encoder/ExpExptbh/vae_encoder/Neg:y:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Exp?
tbh/vae_encoder/Neg_1Neg tbh/vae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Neg_1y
tbh/vae_encoder/Exp_1Exptbh/vae_encoder/Neg_1:y:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Exp_1s
tbh/vae_encoder/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tbh/vae_encoder/add/y?
tbh/vae_encoder/addAddV2tbh/vae_encoder/Exp_1:y:0tbh/vae_encoder/add/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/adds
tbh/vae_encoder/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tbh/vae_encoder/pow/y?
tbh/vae_encoder/powPowtbh/vae_encoder/add:z:0tbh/vae_encoder/pow/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/pow?
tbh/vae_encoder/truedivRealDivtbh/vae_encoder/Exp:y:0tbh/vae_encoder/pow:z:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/truediv?
tbh/vae_encoder/Neg_2Neg"tbh/vae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Neg_2y
tbh/vae_encoder/Exp_2Exptbh/vae_encoder/Neg_2:y:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Exp_2?
tbh/vae_encoder/Neg_3Neg"tbh/vae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Neg_3y
tbh/vae_encoder/Exp_3Exptbh/vae_encoder/Neg_3:y:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Exp_3w
tbh/vae_encoder/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tbh/vae_encoder/add_1/y?
tbh/vae_encoder/add_1AddV2tbh/vae_encoder/Exp_3:y:0 tbh/vae_encoder/add_1/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/add_1w
tbh/vae_encoder/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tbh/vae_encoder/pow_1/y?
tbh/vae_encoder/pow_1Powtbh/vae_encoder/add_1:z:0 tbh/vae_encoder/pow_1/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/pow_1?
tbh/vae_encoder/truediv_1RealDivtbh/vae_encoder/Exp_2:y:0tbh/vae_encoder/pow_1:z:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/truediv_1?
tbh/vae_encoder/mulMultbh/vae_encoder/zeros:output:0"tbh/vae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/mul?
tbh/vae_encoder/add_2AddV2tbh/vae_encoder/mul:z:0 tbh/vae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/add_2x
tbh/vae_encoder/SignSigntbh/vae_encoder/add_2:z:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Signw
tbh/vae_encoder/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tbh/vae_encoder/add_3/y?
tbh/vae_encoder/add_3AddV2tbh/vae_encoder/Sign:y:0 tbh/vae_encoder/add_3/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/add_3
tbh/vae_encoder/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tbh/vae_encoder/truediv_2/y?
tbh/vae_encoder/truediv_2RealDivtbh/vae_encoder/add_3:z:0$tbh/vae_encoder/truediv_2/y:output:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/truediv_2?
tbh/vae_encoder/IdentityIdentitytbh/vae_encoder/truediv_2:z:0*
T0*
_output_shapes

:@2
tbh/vae_encoder/Identity?
tbh/vae_encoder/IdentityN	IdentityNtbh/vae_encoder/truediv_2:z:0 tbh/vae_encoder/Reshape:output:0"tbh/vae_encoder/Reshape_1:output:0tbh/vae_encoder/zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-197195160*<
_output_shapes*
(:@:@:@:@2
tbh/vae_encoder/IdentityN?
-tbh/vae_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp6tbh_vae_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-tbh/vae_encoder/dense_2/MatMul/ReadVariableOp?
tbh/vae_encoder/dense_2/MatMulMatMul(tbh/vae_encoder/dense/Relu:activations:05tbh/vae_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
tbh/vae_encoder/dense_2/MatMul?
.tbh/vae_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp7tbh_vae_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.tbh/vae_encoder/dense_2/BiasAdd/ReadVariableOp?
tbh/vae_encoder/dense_2/BiasAddBiasAdd(tbh/vae_encoder/dense_2/MatMul:product:06tbh/vae_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
tbh/vae_encoder/dense_2/BiasAdd?
tbh/vae_encoder/dense_2/SigmoidSigmoid(tbh/vae_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
tbh/vae_encoder/dense_2/Sigmoid?
IdentityIdentity"tbh/vae_encoder/IdentityN:output:0-^tbh/vae_encoder/dense/BiasAdd/ReadVariableOp/^tbh/vae_encoder/dense/BiasAdd_1/ReadVariableOp,^tbh/vae_encoder/dense/MatMul/ReadVariableOp.^tbh/vae_encoder/dense/MatMul_1/ReadVariableOp/^tbh/vae_encoder/dense_1/BiasAdd/ReadVariableOp.^tbh/vae_encoder/dense_1/MatMul/ReadVariableOp/^tbh/vae_encoder/dense_2/BiasAdd/ReadVariableOp.^tbh/vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:?????????? :?????????
:?????????@:??????????::::::2\
,tbh/vae_encoder/dense/BiasAdd/ReadVariableOp,tbh/vae_encoder/dense/BiasAdd/ReadVariableOp2`
.tbh/vae_encoder/dense/BiasAdd_1/ReadVariableOp.tbh/vae_encoder/dense/BiasAdd_1/ReadVariableOp2Z
+tbh/vae_encoder/dense/MatMul/ReadVariableOp+tbh/vae_encoder/dense/MatMul/ReadVariableOp2^
-tbh/vae_encoder/dense/MatMul_1/ReadVariableOp-tbh/vae_encoder/dense/MatMul_1/ReadVariableOp2`
.tbh/vae_encoder/dense_1/BiasAdd/ReadVariableOp.tbh/vae_encoder/dense_1/BiasAdd/ReadVariableOp2^
-tbh/vae_encoder/dense_1/MatMul/ReadVariableOp-tbh/vae_encoder/dense_1/MatMul/ReadVariableOp2`
.tbh/vae_encoder/dense_2/BiasAdd/ReadVariableOp.tbh/vae_encoder/dense_2/BiasAdd/ReadVariableOp2^
-tbh/vae_encoder/dense_2/MatMul/ReadVariableOp-tbh/vae_encoder/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:?????????? 
#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????

#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
C
$__inference_graph_laplacian_14232069
	adjacency
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceZ

ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2

ones/mul/yi
ones/mulMulstrided_slice:output:0ones/mul/y:output:0*
T0*
_output_shapes
: 2

ones/mul]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/yc
	ones/LessLessones/mul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/Less`
ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/packed/1?
ones/packedPackstrided_slice:output:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Consth
onesFillones/packed:output:0ones/Const:output:0*
T0*
_output_shapes

:2
ones]
matmulMatMul	adjacencyones:output:0*
T0*
_output_shapes

:2
matmulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/y^
addAddV2matmul:product:0add/y:output:0*
T0*
_output_shapes

:2
addS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Pow/yS
PowPowadd:z:0Pow/y:output:0*
T0*
_output_shapes

:2
Powv
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
eye/MinimumY
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
	eye/shapeq
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:2
eye/concat/values_1d
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
eye/concat/axis?

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:2

eye/concate
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
eye/ones/Consto
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes
:2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diagV
mulMuleye/diag:output:0Pow:z:0*
T0*
_output_shapes

:2
mul[
matmul_1MatMulmul:z:0	adjacency*
T0*
_output_shapes

:2

matmul_1d
matmul_2MatMulmatmul_1:product:0mul:z:0*
T0*
_output_shapes

:2

matmul_2]
IdentityIdentitymatmul_2:product:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

::I E

_output_shapes

:
#
_user_specified_name	adjacency
?
?
'__inference_tbh_layer_call_fn_197196400

inputs_0_0

inputs_0_1

inputs_0_2
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1
inputs_0_2inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14* 
Tin
2*
Tout

2*
_collective_manager_ids
 *c
_output_shapesQ
O:@:	? :::?????????:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_tbh_layer_call_and_return_conditional_losses_1971957362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:?????????? :?????????
:?????????@:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:TP
(
_output_shapes
:?????????? 
$
_user_specified_name
inputs/0/1:SO
'
_output_shapes
:?????????

$
_user_specified_name
inputs/0/2:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
?
+__inference_decoder_layer_call_fn_197196680

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_decoder_layer_call_and_return_conditional_losses_1971955612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?G
?
J__inference_vae_encoder_layer_call_and_return_conditional_losses_197195366

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs%dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd_1q
dense/Relu_1Reludense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
dense/Relu_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu_1:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
splito
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape/shapen
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*
_output_shapes

:@2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape_1/shapet
	Reshape_1Reshapesplit:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1c
zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
zerosL
NegNegReshape:output:0*
T0*
_output_shapes

:@2
NegC
ExpExpNeg:y:0*
T0*
_output_shapes

:@2
ExpP
Neg_1NegReshape:output:0*
T0*
_output_shapes

:@2
Neg_1I
Exp_1Exp	Neg_1:y:0*
T0*
_output_shapes

:@2
Exp_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2	Exp_1:y:0add/y:output:0*
T0*
_output_shapes

:@2
addS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yS
powPowadd:z:0pow/y:output:0*
T0*
_output_shapes

:@2
powX
truedivRealDivExp:y:0pow:z:0*
T0*
_output_shapes

:@2	
truedivR
Neg_2NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_2I
Exp_2Exp	Neg_2:y:0*
T0*
_output_shapes

:@2
Exp_2R
Neg_3NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_3I
Exp_3Exp	Neg_3:y:0*
T0*
_output_shapes

:@2
Exp_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_1/y]
add_1AddV2	Exp_3:y:0add_1/y:output:0*
T0*
_output_shapes

:@2
add_1W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y[
pow_1Pow	add_1:z:0pow_1/y:output:0*
T0*
_output_shapes

:@2
pow_1`
	truediv_1RealDiv	Exp_2:y:0	pow_1:z:0*
T0*
_output_shapes

:@2
	truediv_1^
mulMulzeros:output:0Reshape_1:output:0*
T0*
_output_shapes

:@2
mul[
add_2AddV2mul:z:0Reshape:output:0*
T0*
_output_shapes

:@2
add_2H
SignSign	add_2:z:0*
T0*
_output_shapes

:@2
SignW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/y\
add_3AddV2Sign:y:0add_3/y:output:0*
T0*
_output_shapes

:@2
add_3_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yk
	truediv_2RealDiv	add_3:z:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2X
IdentityIdentitytruediv_2:z:0*
T0*
_output_shapes

:@2

Identity?
	IdentityN	IdentityNtruediv_2:z:0Reshape:output:0Reshape_1:output:0zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-197195326*<
_output_shapes*
(:@:@:@:@2
	IdentityN?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Sigmoid?

Identity_1IdentityIdentityN:output:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:?????????? ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_197196704
bbn
cbn
gcn_layer_197196697
gcn_layer_197196699
identity??!gcn_layer/StatefulPartitionedCall?
PartitionedCallPartitionedCallbbn*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_build_adjacency_hamming_142320222
PartitionedCall?
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallcbnPartitionedCall:output:0gcn_layer_197196697gcn_layer_197196699*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_spectrum_conv_142320732#
!gcn_layer/StatefulPartitionedCalls
SigmoidSigmoid*gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2	
Sigmoid{
IdentityIdentitySigmoid:y:0"^gcn_layer/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:@:??????????::2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:MI
(
_output_shapes
:??????????

_user_specified_namecbn
?
?
+__inference_decoder_layer_call_fn_197196667

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_decoder_layer_call_and_return_conditional_losses_1971955432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?	
?
F__inference_dense_6_layer_call_and_return_conditional_losses_197195608

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_tbh_layer_call_fn_197196421

inputs_0_0

inputs_0_1

inputs_0_2
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1
inputs_0_2inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_tbh_layer_call_and_return_conditional_losses_1971958092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:?????????? :?????????
:?????????@:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:TP
(
_output_shapes
:?????????? 
$
_user_specified_name
inputs/0/1:SO
'
_output_shapes
:?????????

$
_user_specified_name
inputs/0/2:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
?
B__inference_tbh_layer_call_and_return_conditional_losses_197195809

inputs
inputs_1
inputs_2
inputs_3
inputs_4
vae_encoder_197195794
vae_encoder_197195796
vae_encoder_197195798
vae_encoder_197195800
vae_encoder_197195802
vae_encoder_197195804
identity??#vae_encoder/StatefulPartitionedCall?
#vae_encoder/StatefulPartitionedCallStatefulPartitionedCallinputs_1vae_encoder_197195794vae_encoder_197195796vae_encoder_197195798vae_encoder_197195800vae_encoder_197195802vae_encoder_197195804*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:@:??????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_vae_encoder_layer_call_and_return_conditional_losses_1971953662%
#vae_encoder/StatefulPartitionedCall?
IdentityIdentity,vae_encoder/StatefulPartitionedCall:output:0$^vae_encoder/StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:?????????? :?????????
:?????????@:??????????::::::2J
#vae_encoder/StatefulPartitionedCall#vae_encoder/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_decoder_layer_call_and_return_conditional_losses_197195561

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_3/BiasAddh
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*
_output_shapes
:	?2
dense_3/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_4/BiasAddh
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*
_output_shapes
:	? 2
dense_4/Relu?
IdentityIdentitydense_4/Relu:activations:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*
_output_shapes
:	? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
"__inference_spectrum_conv_14233577

values
	adjacency*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulvalues%dense_5/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
dense_5/BiasAdd?
PartitionedCallPartitionedCall	adjacency*
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:
??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_graph_laplacian_142335732
PartitionedCally
matmulMatMulPartitionedCall:output:0dense_5/BiasAdd:output:0*
T0* 
_output_shapes
:
??2
matmul?
IdentityIdentitymatmul:product:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :
??:
??::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:H D
 
_output_shapes
:
??
 
_user_specified_namevalues:KG
 
_output_shapes
:
??
#
_user_specified_name	adjacency
?E
?	
%__inference__traced_restore_197196937
file_prefix'
#assignvariableop_tbh_dense_6_kernel'
#assignvariableop_1_tbh_dense_6_bias)
%assignvariableop_2_tbh_dense_7_kernel'
#assignvariableop_3_tbh_dense_7_bias3
/assignvariableop_4_tbh_vae_encoder_dense_kernel1
-assignvariableop_5_tbh_vae_encoder_dense_bias5
1assignvariableop_6_tbh_vae_encoder_dense_1_kernel3
/assignvariableop_7_tbh_vae_encoder_dense_1_bias5
1assignvariableop_8_tbh_vae_encoder_dense_2_kernel3
/assignvariableop_9_tbh_vae_encoder_dense_2_bias2
.assignvariableop_10_tbh_decoder_dense_3_kernel0
,assignvariableop_11_tbh_decoder_dense_3_bias2
.assignvariableop_12_tbh_decoder_dense_4_kernel0
,assignvariableop_13_tbh_decoder_dense_4_bias&
"assignvariableop_14_dense_5_kernel$
 assignvariableop_15_dense_5_bias
identity_17??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'dis_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dis_1/bias/.ATTRIBUTES/VARIABLE_VALUEB'dis_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dis_2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp#assignvariableop_tbh_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_tbh_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp%assignvariableop_2_tbh_dense_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_tbh_dense_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_tbh_vae_encoder_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp-assignvariableop_5_tbh_vae_encoder_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp1assignvariableop_6_tbh_vae_encoder_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_tbh_vae_encoder_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp1assignvariableop_8_tbh_vae_encoder_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_tbh_vae_encoder_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp.assignvariableop_10_tbh_decoder_dense_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_tbh_decoder_dense_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_tbh_decoder_dense_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_tbh_decoder_dense_4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16?
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?P
?
J__inference_vae_encoder_layer_call_and_return_conditional_losses_197195291

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs%dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd_1q
dense/Relu_1Reludense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
dense/Relu_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu_1:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
splito
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape/shapen
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*
_output_shapes

:@2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape_1/shapet
	Reshape_1Reshapesplit:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes

:@2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@2
random_normalw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumrandom_normal:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:@2
clip_by_valueL
NegNegReshape:output:0*
T0*
_output_shapes

:@2
NegC
ExpExpNeg:y:0*
T0*
_output_shapes

:@2
ExpP
Neg_1NegReshape:output:0*
T0*
_output_shapes

:@2
Neg_1I
Exp_1Exp	Neg_1:y:0*
T0*
_output_shapes

:@2
Exp_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2	Exp_1:y:0add/y:output:0*
T0*
_output_shapes

:@2
addS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yS
powPowadd:z:0pow/y:output:0*
T0*
_output_shapes

:@2
powX
truedivRealDivExp:y:0pow:z:0*
T0*
_output_shapes

:@2	
truedivR
Neg_2NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_2I
Exp_2Exp	Neg_2:y:0*
T0*
_output_shapes

:@2
Exp_2R
Neg_3NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_3I
Exp_3Exp	Neg_3:y:0*
T0*
_output_shapes

:@2
Exp_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_1/y]
add_1AddV2	Exp_3:y:0add_1/y:output:0*
T0*
_output_shapes

:@2
add_1W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y[
pow_1Pow	add_1:z:0pow_1/y:output:0*
T0*
_output_shapes

:@2
pow_1`
	truediv_1RealDiv	Exp_2:y:0	pow_1:z:0*
T0*
_output_shapes

:@2
	truediv_1a
mulMulclip_by_value:z:0Reshape_1:output:0*
T0*
_output_shapes

:@2
mul[
add_2AddV2mul:z:0Reshape:output:0*
T0*
_output_shapes

:@2
add_2H
SignSign	add_2:z:0*
T0*
_output_shapes

:@2
SignW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/y\
add_3AddV2Sign:y:0add_3/y:output:0*
T0*
_output_shapes

:@2
add_3_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yk
	truediv_2RealDiv	add_3:z:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2X
IdentityIdentitytruediv_2:z:0*
T0*
_output_shapes

:@2

Identity?
	IdentityN	IdentityNtruediv_2:z:0Reshape:output:0Reshape_1:output:0clip_by_value:z:0*
T
2*/
_gradient_op_typeCustomGradient-197195251*<
_output_shapes*
(:@:@:@:@2
	IdentityN?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Sigmoid?

Identity_1IdentityIdentityN:output:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:?????????? ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
C
$__inference_graph_laplacian_14233573
	adjacency
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceZ

ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2

ones/mul/yi
ones/mulMulstrided_slice:output:0ones/mul/y:output:0*
T0*
_output_shapes
: 2

ones/mul]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/yc
	ones/LessLessones/mul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/Less`
ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/packed/1?
ones/packedPackstrided_slice:output:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Consti
onesFillones/packed:output:0ones/Const:output:0*
T0*
_output_shapes
:	?2
ones^
matmulMatMul	adjacencyones:output:0*
T0*
_output_shapes
:	?2
matmulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/y_
addAddV2matmul:product:0add/y:output:0*
T0*
_output_shapes
:	?2
addS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Pow/yT
PowPowadd:z:0Pow/y:output:0*
T0*
_output_shapes
:	?2
Powv
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
eye/MinimumY
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
	eye/shapeq
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:2
eye/concat/values_1d
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
eye/concat/axis?

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:2

eye/concate
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
eye/ones/Constp
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes	
:?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0* 
_output_shapes
:
??2

eye/diagX
mulMuleye/diag:output:0Pow:z:0*
T0* 
_output_shapes
:
??2
mul]
matmul_1MatMulmul:z:0	adjacency*
T0* 
_output_shapes
:
??2

matmul_1f
matmul_2MatMulmatmul_1:product:0mul:z:0*
T0* 
_output_shapes
:
??2

matmul_2_
IdentityIdentitymatmul_2:product:0*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*
_input_shapes
:
??:K G
 
_output_shapes
:
??
#
_user_specified_name	adjacency
?\
?
B__inference_tbh_layer_call_and_return_conditional_losses_197196349

inputs_0_0

inputs_0_1

inputs_0_2
inputs_1
inputs_24
0vae_encoder_dense_matmul_readvariableop_resource5
1vae_encoder_dense_biasadd_readvariableop_resource6
2vae_encoder_dense_1_matmul_readvariableop_resource7
3vae_encoder_dense_1_biasadd_readvariableop_resource6
2vae_encoder_dense_2_matmul_readvariableop_resource7
3vae_encoder_dense_2_biasadd_readvariableop_resource
identity??(vae_encoder/dense/BiasAdd/ReadVariableOp?*vae_encoder/dense/BiasAdd_1/ReadVariableOp?'vae_encoder/dense/MatMul/ReadVariableOp?)vae_encoder/dense/MatMul_1/ReadVariableOp?*vae_encoder/dense_1/BiasAdd/ReadVariableOp?)vae_encoder/dense_1/MatMul/ReadVariableOp?*vae_encoder/dense_2/BiasAdd/ReadVariableOp?)vae_encoder/dense_2/MatMul/ReadVariableOp`
vae_encoder/ShapeShape
inputs_0_1*
T0*
_output_shapes
:2
vae_encoder/Shape?
vae_encoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
vae_encoder/strided_slice/stack?
!vae_encoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!vae_encoder/strided_slice/stack_1?
!vae_encoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!vae_encoder/strided_slice/stack_2?
vae_encoder/strided_sliceStridedSlicevae_encoder/Shape:output:0(vae_encoder/strided_slice/stack:output:0*vae_encoder/strided_slice/stack_1:output:0*vae_encoder/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vae_encoder/strided_slice?
'vae_encoder/dense/MatMul/ReadVariableOpReadVariableOp0vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02)
'vae_encoder/dense/MatMul/ReadVariableOp?
vae_encoder/dense/MatMulMatMul
inputs_0_1/vae_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/MatMul?
(vae_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp1vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(vae_encoder/dense/BiasAdd/ReadVariableOp?
vae_encoder/dense/BiasAddBiasAdd"vae_encoder/dense/MatMul:product:00vae_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/BiasAdd?
vae_encoder/dense/ReluRelu"vae_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/Relu?
)vae_encoder/dense/MatMul_1/ReadVariableOpReadVariableOp0vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02+
)vae_encoder/dense/MatMul_1/ReadVariableOp?
vae_encoder/dense/MatMul_1MatMul
inputs_0_11vae_encoder/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/MatMul_1?
*vae_encoder/dense/BiasAdd_1/ReadVariableOpReadVariableOp1vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense/BiasAdd_1/ReadVariableOp?
vae_encoder/dense/BiasAdd_1BiasAdd$vae_encoder/dense/MatMul_1:product:02vae_encoder/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/BiasAdd_1?
vae_encoder/dense/Relu_1Relu$vae_encoder/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/Relu_1?
)vae_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)vae_encoder/dense_1/MatMul/ReadVariableOp?
vae_encoder/dense_1/MatMulMatMul&vae_encoder/dense/Relu_1:activations:01vae_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_1/MatMul?
*vae_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense_1/BiasAdd/ReadVariableOp?
vae_encoder/dense_1/BiasAddBiasAdd$vae_encoder/dense_1/MatMul:product:02vae_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_1/BiasAddh
vae_encoder/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder/Const|
vae_encoder/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder/split/split_dim?
vae_encoder/splitSplit$vae_encoder/split/split_dim:output:0$vae_encoder/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
vae_encoder/split?
vae_encoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
vae_encoder/Reshape/shape?
vae_encoder/ReshapeReshapevae_encoder/split:output:0"vae_encoder/Reshape/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Reshape?
vae_encoder/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
vae_encoder/Reshape_1/shape?
vae_encoder/Reshape_1Reshapevae_encoder/split:output:1$vae_encoder/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Reshape_1{
vae_encoder/zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
vae_encoder/zerosp
vae_encoder/NegNegvae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Negg
vae_encoder/ExpExpvae_encoder/Neg:y:0*
T0*
_output_shapes

:@2
vae_encoder/Expt
vae_encoder/Neg_1Negvae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_1m
vae_encoder/Exp_1Expvae_encoder/Neg_1:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_1k
vae_encoder/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add/y?
vae_encoder/addAddV2vae_encoder/Exp_1:y:0vae_encoder/add/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/addk
vae_encoder/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/pow/y?
vae_encoder/powPowvae_encoder/add:z:0vae_encoder/pow/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/pow?
vae_encoder/truedivRealDivvae_encoder/Exp:y:0vae_encoder/pow:z:0*
T0*
_output_shapes

:@2
vae_encoder/truedivv
vae_encoder/Neg_2Negvae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_2m
vae_encoder/Exp_2Expvae_encoder/Neg_2:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_2v
vae_encoder/Neg_3Negvae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_3m
vae_encoder/Exp_3Expvae_encoder/Neg_3:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_3o
vae_encoder/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add_1/y?
vae_encoder/add_1AddV2vae_encoder/Exp_3:y:0vae_encoder/add_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_1o
vae_encoder/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/pow_1/y?
vae_encoder/pow_1Powvae_encoder/add_1:z:0vae_encoder/pow_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/pow_1?
vae_encoder/truediv_1RealDivvae_encoder/Exp_2:y:0vae_encoder/pow_1:z:0*
T0*
_output_shapes

:@2
vae_encoder/truediv_1?
vae_encoder/mulMulvae_encoder/zeros:output:0vae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/mul?
vae_encoder/add_2AddV2vae_encoder/mul:z:0vae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_2l
vae_encoder/SignSignvae_encoder/add_2:z:0*
T0*
_output_shapes

:@2
vae_encoder/Signo
vae_encoder/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add_3/y?
vae_encoder/add_3AddV2vae_encoder/Sign:y:0vae_encoder/add_3/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_3w
vae_encoder/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/truediv_2/y?
vae_encoder/truediv_2RealDivvae_encoder/add_3:z:0 vae_encoder/truediv_2/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/truediv_2|
vae_encoder/IdentityIdentityvae_encoder/truediv_2:z:0*
T0*
_output_shapes

:@2
vae_encoder/Identity?
vae_encoder/IdentityN	IdentityNvae_encoder/truediv_2:z:0vae_encoder/Reshape:output:0vae_encoder/Reshape_1:output:0vae_encoder/zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-197196310*<
_output_shapes*
(:@:@:@:@2
vae_encoder/IdentityN?
)vae_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)vae_encoder/dense_2/MatMul/ReadVariableOp?
vae_encoder/dense_2/MatMulMatMul$vae_encoder/dense/Relu:activations:01vae_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/MatMul?
*vae_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense_2/BiasAdd/ReadVariableOp?
vae_encoder/dense_2/BiasAddBiasAdd$vae_encoder/dense_2/MatMul:product:02vae_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/BiasAdd?
vae_encoder/dense_2/SigmoidSigmoid$vae_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/Sigmoid?
IdentityIdentityvae_encoder/IdentityN:output:0)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:?????????? :?????????
:?????????@:??????????::::::2T
(vae_encoder/dense/BiasAdd/ReadVariableOp(vae_encoder/dense/BiasAdd/ReadVariableOp2X
*vae_encoder/dense/BiasAdd_1/ReadVariableOp*vae_encoder/dense/BiasAdd_1/ReadVariableOp2R
'vae_encoder/dense/MatMul/ReadVariableOp'vae_encoder/dense/MatMul/ReadVariableOp2V
)vae_encoder/dense/MatMul_1/ReadVariableOp)vae_encoder/dense/MatMul_1/ReadVariableOp2X
*vae_encoder/dense_1/BiasAdd/ReadVariableOp*vae_encoder/dense_1/BiasAdd/ReadVariableOp2V
)vae_encoder/dense_1/MatMul/ReadVariableOp)vae_encoder/dense_1/MatMul/ReadVariableOp2X
*vae_encoder/dense_2/BiasAdd/ReadVariableOp*vae_encoder/dense_2/BiasAdd/ReadVariableOp2V
)vae_encoder/dense_2/MatMul/ReadVariableOp)vae_encoder/dense_2/MatMul/ReadVariableOp:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:TP
(
_output_shapes
:?????????? 
$
_user_specified_name
inputs/0/1:SO
'
_output_shapes
:?????????

$
_user_specified_name
inputs/0/2:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
?
+__inference_dense_6_layer_call_fn_197196764

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_6_layer_call_and_return_conditional_losses_1971956082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
B__inference_tbh_layer_call_and_return_conditional_losses_197195984
	input_1_1
	input_1_2
	input_1_3
input_2
input_34
0vae_encoder_dense_matmul_readvariableop_resource5
1vae_encoder_dense_biasadd_readvariableop_resource6
2vae_encoder_dense_1_matmul_readvariableop_resource7
3vae_encoder_dense_1_biasadd_readvariableop_resource6
2vae_encoder_dense_2_matmul_readvariableop_resource7
3vae_encoder_dense_2_biasadd_readvariableop_resource'
#twin_bottleneck_gcn_layer_197195934'
#twin_bottleneck_gcn_layer_197195936*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource2
.decoder_dense_3_matmul_readvariableop_resource3
/decoder_dense_3_biasadd_readvariableop_resource2
.decoder_dense_4_matmul_readvariableop_resource3
/decoder_dense_4_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5??&decoder/dense_3/BiasAdd/ReadVariableOp?%decoder/dense_3/MatMul/ReadVariableOp?&decoder/dense_4/BiasAdd/ReadVariableOp?%decoder/dense_4/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp? dense_6/BiasAdd_1/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_6/MatMul_1/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp? dense_7/BiasAdd_1/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_7/MatMul_1/ReadVariableOp?1twin_bottleneck/gcn_layer/StatefulPartitionedCall?(vae_encoder/dense/BiasAdd/ReadVariableOp?*vae_encoder/dense/BiasAdd_1/ReadVariableOp?'vae_encoder/dense/MatMul/ReadVariableOp?)vae_encoder/dense/MatMul_1/ReadVariableOp?*vae_encoder/dense_1/BiasAdd/ReadVariableOp?)vae_encoder/dense_1/MatMul/ReadVariableOp?*vae_encoder/dense_2/BiasAdd/ReadVariableOp?)vae_encoder/dense_2/MatMul/ReadVariableOp_
vae_encoder/ShapeShape	input_1_2*
T0*
_output_shapes
:2
vae_encoder/Shape?
vae_encoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
vae_encoder/strided_slice/stack?
!vae_encoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!vae_encoder/strided_slice/stack_1?
!vae_encoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!vae_encoder/strided_slice/stack_2?
vae_encoder/strided_sliceStridedSlicevae_encoder/Shape:output:0(vae_encoder/strided_slice/stack:output:0*vae_encoder/strided_slice/stack_1:output:0*vae_encoder/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vae_encoder/strided_slice?
'vae_encoder/dense/MatMul/ReadVariableOpReadVariableOp0vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02)
'vae_encoder/dense/MatMul/ReadVariableOp?
vae_encoder/dense/MatMulMatMul	input_1_2/vae_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/MatMul?
(vae_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp1vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(vae_encoder/dense/BiasAdd/ReadVariableOp?
vae_encoder/dense/BiasAddBiasAdd"vae_encoder/dense/MatMul:product:00vae_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/BiasAdd?
vae_encoder/dense/ReluRelu"vae_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/Relu?
)vae_encoder/dense/MatMul_1/ReadVariableOpReadVariableOp0vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02+
)vae_encoder/dense/MatMul_1/ReadVariableOp?
vae_encoder/dense/MatMul_1MatMul	input_1_21vae_encoder/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/MatMul_1?
*vae_encoder/dense/BiasAdd_1/ReadVariableOpReadVariableOp1vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense/BiasAdd_1/ReadVariableOp?
vae_encoder/dense/BiasAdd_1BiasAdd$vae_encoder/dense/MatMul_1:product:02vae_encoder/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/BiasAdd_1?
vae_encoder/dense/Relu_1Relu$vae_encoder/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/Relu_1?
)vae_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)vae_encoder/dense_1/MatMul/ReadVariableOp?
vae_encoder/dense_1/MatMulMatMul&vae_encoder/dense/Relu_1:activations:01vae_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_1/MatMul?
*vae_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense_1/BiasAdd/ReadVariableOp?
vae_encoder/dense_1/BiasAddBiasAdd$vae_encoder/dense_1/MatMul:product:02vae_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_1/BiasAddh
vae_encoder/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder/Const|
vae_encoder/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder/split/split_dim?
vae_encoder/splitSplit$vae_encoder/split/split_dim:output:0$vae_encoder/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
vae_encoder/split?
vae_encoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
vae_encoder/Reshape/shape?
vae_encoder/ReshapeReshapevae_encoder/split:output:0"vae_encoder/Reshape/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Reshape?
vae_encoder/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
vae_encoder/Reshape_1/shape?
vae_encoder/Reshape_1Reshapevae_encoder/split:output:1$vae_encoder/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Reshape_1?
vae_encoder/random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2!
vae_encoder/random_normal/shape?
vae_encoder/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
vae_encoder/random_normal/mean?
 vae_encoder/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 vae_encoder/random_normal/stddev?
.vae_encoder/random_normal/RandomStandardNormalRandomStandardNormal(vae_encoder/random_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype020
.vae_encoder/random_normal/RandomStandardNormal?
vae_encoder/random_normal/mulMul7vae_encoder/random_normal/RandomStandardNormal:output:0)vae_encoder/random_normal/stddev:output:0*
T0*
_output_shapes

:@2
vae_encoder/random_normal/mul?
vae_encoder/random_normalAdd!vae_encoder/random_normal/mul:z:0'vae_encoder/random_normal/mean:output:0*
T0*
_output_shapes

:@2
vae_encoder/random_normal?
#vae_encoder/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#vae_encoder/clip_by_value/Minimum/y?
!vae_encoder/clip_by_value/MinimumMinimumvae_encoder/random_normal:z:0,vae_encoder/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@2#
!vae_encoder/clip_by_value/Minimum
vae_encoder/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/clip_by_value/y?
vae_encoder/clip_by_valueMaximum%vae_encoder/clip_by_value/Minimum:z:0$vae_encoder/clip_by_value/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/clip_by_valuep
vae_encoder/NegNegvae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Negg
vae_encoder/ExpExpvae_encoder/Neg:y:0*
T0*
_output_shapes

:@2
vae_encoder/Expt
vae_encoder/Neg_1Negvae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_1m
vae_encoder/Exp_1Expvae_encoder/Neg_1:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_1k
vae_encoder/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add/y?
vae_encoder/addAddV2vae_encoder/Exp_1:y:0vae_encoder/add/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/addk
vae_encoder/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/pow/y?
vae_encoder/powPowvae_encoder/add:z:0vae_encoder/pow/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/pow?
vae_encoder/truedivRealDivvae_encoder/Exp:y:0vae_encoder/pow:z:0*
T0*
_output_shapes

:@2
vae_encoder/truedivv
vae_encoder/Neg_2Negvae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_2m
vae_encoder/Exp_2Expvae_encoder/Neg_2:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_2v
vae_encoder/Neg_3Negvae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_3m
vae_encoder/Exp_3Expvae_encoder/Neg_3:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_3o
vae_encoder/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add_1/y?
vae_encoder/add_1AddV2vae_encoder/Exp_3:y:0vae_encoder/add_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_1o
vae_encoder/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/pow_1/y?
vae_encoder/pow_1Powvae_encoder/add_1:z:0vae_encoder/pow_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/pow_1?
vae_encoder/truediv_1RealDivvae_encoder/Exp_2:y:0vae_encoder/pow_1:z:0*
T0*
_output_shapes

:@2
vae_encoder/truediv_1?
vae_encoder/mulMulvae_encoder/clip_by_value:z:0vae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/mul?
vae_encoder/add_2AddV2vae_encoder/mul:z:0vae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_2l
vae_encoder/SignSignvae_encoder/add_2:z:0*
T0*
_output_shapes

:@2
vae_encoder/Signo
vae_encoder/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add_3/y?
vae_encoder/add_3AddV2vae_encoder/Sign:y:0vae_encoder/add_3/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_3w
vae_encoder/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/truediv_2/y?
vae_encoder/truediv_2RealDivvae_encoder/add_3:z:0 vae_encoder/truediv_2/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/truediv_2|
vae_encoder/IdentityIdentityvae_encoder/truediv_2:z:0*
T0*
_output_shapes

:@2
vae_encoder/Identity?
vae_encoder/IdentityN	IdentityNvae_encoder/truediv_2:z:0vae_encoder/Reshape:output:0vae_encoder/Reshape_1:output:0vae_encoder/clip_by_value:z:0*
T
2*/
_gradient_op_typeCustomGradient-197195895*<
_output_shapes*
(:@:@:@:@2
vae_encoder/IdentityN?
)vae_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)vae_encoder/dense_2/MatMul/ReadVariableOp?
vae_encoder/dense_2/MatMulMatMul$vae_encoder/dense/Relu:activations:01vae_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/MatMul?
*vae_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense_2/BiasAdd/ReadVariableOp?
vae_encoder/dense_2/BiasAddBiasAdd$vae_encoder/dense_2/MatMul:product:02vae_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/BiasAdd?
vae_encoder/dense_2/SigmoidSigmoid$vae_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/Sigmoid?
twin_bottleneck/PartitionedCallPartitionedCallvae_encoder/IdentityN:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_build_adjacency_hamming_142320222!
twin_bottleneck/PartitionedCall?
1twin_bottleneck/gcn_layer/StatefulPartitionedCallStatefulPartitionedCallvae_encoder/dense_2/Sigmoid:y:0(twin_bottleneck/PartitionedCall:output:0#twin_bottleneck_gcn_layer_197195934#twin_bottleneck_gcn_layer_197195936*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_spectrum_conv_1423207323
1twin_bottleneck/gcn_layer/StatefulPartitionedCall?
twin_bottleneck/SigmoidSigmoid:twin_bottleneck/gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2
twin_bottleneck/Sigmoid?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulvae_encoder/IdentityN:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_6/BiasAddp
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_6/Sigmoid?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMultwin_bottleneck/Sigmoid:y:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_7/BiasAddp
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_7/Sigmoid?
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%decoder/dense_3/MatMul/ReadVariableOp?
decoder/dense_3/MatMulMatMultwin_bottleneck/Sigmoid:y:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
decoder/dense_3/MatMul?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_3/BiasAdd/ReadVariableOp?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
decoder/dense_3/BiasAdd?
decoder/dense_3/ReluRelu decoder/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:	?2
decoder/dense_3/Relu?
%decoder/dense_4/MatMul/ReadVariableOpReadVariableOp.decoder_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02'
%decoder/dense_4/MatMul/ReadVariableOp?
decoder/dense_4/MatMulMatMul"decoder/dense_3/Relu:activations:0-decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
decoder/dense_4/MatMul?
&decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02(
&decoder/dense_4/BiasAdd/ReadVariableOp?
decoder/dense_4/BiasAddBiasAdd decoder/dense_4/MatMul:product:0.decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
decoder/dense_4/BiasAdd?
decoder/dense_4/ReluRelu decoder/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:	? 2
decoder/dense_4/Relu?
dense_6/MatMul_1/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_6/MatMul_1/ReadVariableOp?
dense_6/MatMul_1MatMulinput_2'dense_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul_1?
 dense_6/BiasAdd_1/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_6/BiasAdd_1/ReadVariableOp?
dense_6/BiasAdd_1BiasAdddense_6/MatMul_1:product:0(dense_6/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAdd_1
dense_6/Sigmoid_1Sigmoiddense_6/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
dense_6/Sigmoid_1?
dense_7/MatMul_1/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_7/MatMul_1/ReadVariableOp?
dense_7/MatMul_1MatMulinput_3'dense_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul_1?
 dense_7/BiasAdd_1/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_7/BiasAdd_1/ReadVariableOp?
dense_7/BiasAdd_1BiasAdddense_7/MatMul_1:product:0(dense_7/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAdd_1
dense_7/Sigmoid_1Sigmoiddense_7/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
dense_7/Sigmoid_1?
IdentityIdentityvae_encoder/IdentityN:output:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity"decoder/dense_4/Relu:activations:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	? 2

Identity_1?

Identity_2Identitydense_6/Sigmoid:y:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identitydense_7/Sigmoid:y:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identitydense_6/Sigmoid_1:y:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_7/Sigmoid_1:y:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:?????????? :?????????
:?????????@:??????????::::::::::::::::2P
&decoder/dense_3/BiasAdd/ReadVariableOp&decoder/dense_3/BiasAdd/ReadVariableOp2N
%decoder/dense_3/MatMul/ReadVariableOp%decoder/dense_3/MatMul/ReadVariableOp2P
&decoder/dense_4/BiasAdd/ReadVariableOp&decoder/dense_4/BiasAdd/ReadVariableOp2N
%decoder/dense_4/MatMul/ReadVariableOp%decoder/dense_4/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/BiasAdd_1/ReadVariableOp dense_6/BiasAdd_1/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2B
dense_6/MatMul_1/ReadVariableOpdense_6/MatMul_1/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/BiasAdd_1/ReadVariableOp dense_7/BiasAdd_1/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2B
dense_7/MatMul_1/ReadVariableOpdense_7/MatMul_1/ReadVariableOp2f
1twin_bottleneck/gcn_layer/StatefulPartitionedCall1twin_bottleneck/gcn_layer/StatefulPartitionedCall2T
(vae_encoder/dense/BiasAdd/ReadVariableOp(vae_encoder/dense/BiasAdd/ReadVariableOp2X
*vae_encoder/dense/BiasAdd_1/ReadVariableOp*vae_encoder/dense/BiasAdd_1/ReadVariableOp2R
'vae_encoder/dense/MatMul/ReadVariableOp'vae_encoder/dense/MatMul/ReadVariableOp2V
)vae_encoder/dense/MatMul_1/ReadVariableOp)vae_encoder/dense/MatMul_1/ReadVariableOp2X
*vae_encoder/dense_1/BiasAdd/ReadVariableOp*vae_encoder/dense_1/BiasAdd/ReadVariableOp2V
)vae_encoder/dense_1/MatMul/ReadVariableOp)vae_encoder/dense_1/MatMul/ReadVariableOp2X
*vae_encoder/dense_2/BiasAdd/ReadVariableOp*vae_encoder/dense_2/BiasAdd/ReadVariableOp2V
)vae_encoder/dense_2/MatMul/ReadVariableOp)vae_encoder/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:?????????? 
#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????

#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_197195444
bbn
cbn
gcn_layer_197195437
gcn_layer_197195439
identity??!gcn_layer/StatefulPartitionedCall?
PartitionedCallPartitionedCallbbn*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_build_adjacency_hamming_142320222
PartitionedCall?
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallcbnPartitionedCall:output:0gcn_layer_197195437gcn_layer_197195439*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_spectrum_conv_142320732#
!gcn_layer/StatefulPartitionedCalls
SigmoidSigmoid*gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2	
Sigmoid{
IdentityIdentitySigmoid:y:0"^gcn_layer/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:@:??????????::2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:MI
(
_output_shapes
:??????????

_user_specified_namecbn
?G
?
J__inference_vae_encoder_layer_call_and_return_conditional_losses_197196580

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs%dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd_1q
dense/Relu_1Reludense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
dense/Relu_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu_1:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
splito
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape/shapen
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*
_output_shapes

:@2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape_1/shapet
	Reshape_1Reshapesplit:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1c
zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
zerosL
NegNegReshape:output:0*
T0*
_output_shapes

:@2
NegC
ExpExpNeg:y:0*
T0*
_output_shapes

:@2
ExpP
Neg_1NegReshape:output:0*
T0*
_output_shapes

:@2
Neg_1I
Exp_1Exp	Neg_1:y:0*
T0*
_output_shapes

:@2
Exp_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2	Exp_1:y:0add/y:output:0*
T0*
_output_shapes

:@2
addS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yS
powPowadd:z:0pow/y:output:0*
T0*
_output_shapes

:@2
powX
truedivRealDivExp:y:0pow:z:0*
T0*
_output_shapes

:@2	
truedivR
Neg_2NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_2I
Exp_2Exp	Neg_2:y:0*
T0*
_output_shapes

:@2
Exp_2R
Neg_3NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_3I
Exp_3Exp	Neg_3:y:0*
T0*
_output_shapes

:@2
Exp_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_1/y]
add_1AddV2	Exp_3:y:0add_1/y:output:0*
T0*
_output_shapes

:@2
add_1W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y[
pow_1Pow	add_1:z:0pow_1/y:output:0*
T0*
_output_shapes

:@2
pow_1`
	truediv_1RealDiv	Exp_2:y:0	pow_1:z:0*
T0*
_output_shapes

:@2
	truediv_1^
mulMulzeros:output:0Reshape_1:output:0*
T0*
_output_shapes

:@2
mul[
add_2AddV2mul:z:0Reshape:output:0*
T0*
_output_shapes

:@2
add_2H
SignSign	add_2:z:0*
T0*
_output_shapes

:@2
SignW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/y\
add_3AddV2Sign:y:0add_3/y:output:0*
T0*
_output_shapes

:@2
add_3_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yk
	truediv_2RealDiv	add_3:z:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2X
IdentityIdentitytruediv_2:z:0*
T0*
_output_shapes

:@2

Identity?
	IdentityN	IdentityNtruediv_2:z:0Reshape:output:0Reshape_1:output:0zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-197196540*<
_output_shapes*
(:@:@:@:@2
	IdentityN?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Sigmoid?

Identity_1IdentityIdentityN:output:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:?????????? ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?	
?
F__inference_dense_6_layer_call_and_return_conditional_losses_197196735

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:@
 
_user_specified_nameinputs
?	
?
/__inference_vae_encoder_layer_call_fn_197196618

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:@:??????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_vae_encoder_layer_call_and_return_conditional_losses_1971953662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:?????????? ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
3__inference_twin_bottleneck_layer_call_fn_197196724
bbn
cbn
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbbncbnunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_1971954442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:@:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:MI
(
_output_shapes
:??????????

_user_specified_namecbn
?P
?
J__inference_vae_encoder_layer_call_and_return_conditional_losses_197196505

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs%dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd_1q
dense/Relu_1Reludense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
dense/Relu_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu_1:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
splito
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape/shapen
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*
_output_shapes

:@2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Reshape_1/shapet
	Reshape_1Reshapesplit:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype02$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes

:@2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@2
random_normalw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimumrandom_normal:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:@2
clip_by_valueL
NegNegReshape:output:0*
T0*
_output_shapes

:@2
NegC
ExpExpNeg:y:0*
T0*
_output_shapes

:@2
ExpP
Neg_1NegReshape:output:0*
T0*
_output_shapes

:@2
Neg_1I
Exp_1Exp	Neg_1:y:0*
T0*
_output_shapes

:@2
Exp_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
add/yW
addAddV2	Exp_1:y:0add/y:output:0*
T0*
_output_shapes

:@2
addS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yS
powPowadd:z:0pow/y:output:0*
T0*
_output_shapes

:@2
powX
truedivRealDivExp:y:0pow:z:0*
T0*
_output_shapes

:@2	
truedivR
Neg_2NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_2I
Exp_2Exp	Neg_2:y:0*
T0*
_output_shapes

:@2
Exp_2R
Neg_3NegReshape_1:output:0*
T0*
_output_shapes

:@2
Neg_3I
Exp_3Exp	Neg_3:y:0*
T0*
_output_shapes

:@2
Exp_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_1/y]
add_1AddV2	Exp_3:y:0add_1/y:output:0*
T0*
_output_shapes

:@2
add_1W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/y[
pow_1Pow	add_1:z:0pow_1/y:output:0*
T0*
_output_shapes

:@2
pow_1`
	truediv_1RealDiv	Exp_2:y:0	pow_1:z:0*
T0*
_output_shapes

:@2
	truediv_1a
mulMulclip_by_value:z:0Reshape_1:output:0*
T0*
_output_shapes

:@2
mul[
add_2AddV2mul:z:0Reshape:output:0*
T0*
_output_shapes

:@2
add_2H
SignSign	add_2:z:0*
T0*
_output_shapes

:@2
SignW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
add_3/y\
add_3AddV2Sign:y:0add_3/y:output:0*
T0*
_output_shapes

:@2
add_3_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yk
	truediv_2RealDiv	add_3:z:0truediv_2/y:output:0*
T0*
_output_shapes

:@2
	truediv_2X
IdentityIdentitytruediv_2:z:0*
T0*
_output_shapes

:@2

Identity?
	IdentityN	IdentityNtruediv_2:z:0Reshape:output:0Reshape_1:output:0clip_by_value:z:0*
T
2*/
_gradient_op_typeCustomGradient-197196465*<
_output_shapes*
(:@:@:@:@2
	IdentityN?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Sigmoid?

Identity_1IdentityIdentityN:output:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity_1?

Identity_2Identitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:?????????? ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?	
?
F__inference_dense_6_layer_call_and_return_conditional_losses_197196755

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
F__inference_dense_7_layer_call_and_return_conditional_losses_197195631

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_decoder_layer_call_and_return_conditional_losses_197196636

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_3/BiasAddh
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*
_output_shapes
:	?2
dense_3/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_4/BiasAddh
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*
_output_shapes
:	? 2
dense_4/Relu?
IdentityIdentitydense_4/Relu:activations:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*
_output_shapes
:	? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
C
$__inference_graph_laplacian_14233490
	adjacency
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceZ

ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2

ones/mul/yi
ones/mulMulstrided_slice:output:0ones/mul/y:output:0*
T0*
_output_shapes
: 2

ones/mul]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/yc
	ones/LessLessones/mul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/Less`
ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/packed/1?
ones/packedPackstrided_slice:output:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Consti
onesFillones/packed:output:0ones/Const:output:0*
T0*
_output_shapes
:	?2
ones^
matmulMatMul	adjacencyones:output:0*
T0*
_output_shapes
:	?2
matmulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/y_
addAddV2matmul:product:0add/y:output:0*
T0*
_output_shapes
:	?2
addS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Pow/yT
PowPowadd:z:0Pow/y:output:0*
T0*
_output_shapes
:	?2
Powv
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
eye/MinimumY
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
	eye/shapeq
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:2
eye/concat/values_1d
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
eye/concat/axis?

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:2

eye/concate
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
eye/ones/Constp
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes	
:?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0* 
_output_shapes
:
??2

eye/diagX
mulMuleye/diag:output:0Pow:z:0*
T0* 
_output_shapes
:
??2
mul]
matmul_1MatMulmul:z:0	adjacency*
T0* 
_output_shapes
:
??2

matmul_1f
matmul_2MatMulmatmul_1:product:0mul:z:0*
T0* 
_output_shapes
:
??2

matmul_2_
IdentityIdentitymatmul_2:product:0*
T0* 
_output_shapes
:
??2

Identity"
identityIdentity:output:0*
_input_shapes
:
??:K G
 
_output_shapes
:
??
#
_user_specified_name	adjacency
?	
?
F__inference_dense_7_layer_call_and_return_conditional_losses_197195509

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
"__inference_spectrum_conv_14232073

values
	adjacency*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulvalues%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAdd?
PartitionedCallPartitionedCall	adjacency*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_graph_laplacian_142320692
PartitionedCallx
matmulMatMulPartitionedCall:output:0dense_5/BiasAdd:output:0*
T0*
_output_shapes
:	?2
matmul?
IdentityIdentitymatmul:product:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:??????????:::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_namevalues:IE

_output_shapes

:
#
_user_specified_name	adjacency
?	
?
F__inference_dense_7_layer_call_and_return_conditional_losses_197196795

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
'__inference_tbh_layer_call_fn_197196113
	input_1_1
	input_1_2
	input_1_3
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_1_1	input_1_2	input_1_3input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14* 
Tin
2*
Tout

2*
_collective_manager_ids
 *c
_output_shapesQ
O:@:	? :::?????????:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_tbh_layer_call_and_return_conditional_losses_1971957362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:?????????? :?????????
:?????????@:??????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:?????????? 
#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????

#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?+
?
"__inference__traced_save_197196879
file_prefix1
-savev2_tbh_dense_6_kernel_read_readvariableop/
+savev2_tbh_dense_6_bias_read_readvariableop1
-savev2_tbh_dense_7_kernel_read_readvariableop/
+savev2_tbh_dense_7_bias_read_readvariableop;
7savev2_tbh_vae_encoder_dense_kernel_read_readvariableop9
5savev2_tbh_vae_encoder_dense_bias_read_readvariableop=
9savev2_tbh_vae_encoder_dense_1_kernel_read_readvariableop;
7savev2_tbh_vae_encoder_dense_1_bias_read_readvariableop=
9savev2_tbh_vae_encoder_dense_2_kernel_read_readvariableop;
7savev2_tbh_vae_encoder_dense_2_bias_read_readvariableop9
5savev2_tbh_decoder_dense_3_kernel_read_readvariableop7
3savev2_tbh_decoder_dense_3_bias_read_readvariableop9
5savev2_tbh_decoder_dense_4_kernel_read_readvariableop7
3savev2_tbh_decoder_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'dis_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dis_1/bias/.ATTRIBUTES/VARIABLE_VALUEB'dis_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dis_2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_tbh_dense_6_kernel_read_readvariableop+savev2_tbh_dense_6_bias_read_readvariableop-savev2_tbh_dense_7_kernel_read_readvariableop+savev2_tbh_dense_7_bias_read_readvariableop7savev2_tbh_vae_encoder_dense_kernel_read_readvariableop5savev2_tbh_vae_encoder_dense_bias_read_readvariableop9savev2_tbh_vae_encoder_dense_1_kernel_read_readvariableop7savev2_tbh_vae_encoder_dense_1_bias_read_readvariableop9savev2_tbh_vae_encoder_dense_2_kernel_read_readvariableop7savev2_tbh_vae_encoder_dense_2_bias_read_readvariableop5savev2_tbh_decoder_dense_3_kernel_read_readvariableop3savev2_tbh_decoder_dense_3_bias_read_readvariableop5savev2_tbh_decoder_dense_4_kernel_read_readvariableop3savev2_tbh_decoder_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@::	?::
? ?:?:
??:?:
??:?:
??:?:
?? :? :
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
? ?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
?? :!

_output_shapes	
:? :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?	
?
F__inference_dense_7_layer_call_and_return_conditional_losses_197196775

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_197196692
bbn
cbn
gcn_layer_197196685
gcn_layer_197196687
identity??!gcn_layer/StatefulPartitionedCall?
PartitionedCallPartitionedCallbbn*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_build_adjacency_hamming_142320222
PartitionedCall?
!gcn_layer/StatefulPartitionedCallStatefulPartitionedCallcbnPartitionedCall:output:0gcn_layer_197196685gcn_layer_197196687*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_spectrum_conv_142320732#
!gcn_layer/StatefulPartitionedCalls
SigmoidSigmoid*gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2	
Sigmoid{
IdentityIdentitySigmoid:y:0"^gcn_layer/StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:@:??????????::2F
!gcn_layer/StatefulPartitionedCall!gcn_layer/StatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:MI
(
_output_shapes
:??????????

_user_specified_namecbn
?<
?
B__inference_tbh_layer_call_and_return_conditional_losses_197195736

inputs
inputs_1
inputs_2
inputs_3
inputs_4
vae_encoder_197195686
vae_encoder_197195688
vae_encoder_197195690
vae_encoder_197195692
vae_encoder_197195694
vae_encoder_197195696
twin_bottleneck_197195700
twin_bottleneck_197195702
dense_6_197195705
dense_6_197195707
dense_7_197195710
dense_7_197195712
decoder_197195715
decoder_197195717
decoder_197195719
decoder_197195721
identity

identity_1

identity_2

identity_3

identity_4

identity_5??decoder/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?!dense_6/StatefulPartitionedCall_1?dense_7/StatefulPartitionedCall?!dense_7/StatefulPartitionedCall_1?'twin_bottleneck/StatefulPartitionedCall?#vae_encoder/StatefulPartitionedCall?
#vae_encoder/StatefulPartitionedCallStatefulPartitionedCallinputs_1vae_encoder_197195686vae_encoder_197195688vae_encoder_197195690vae_encoder_197195692vae_encoder_197195694vae_encoder_197195696*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:@:??????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_vae_encoder_layer_call_and_return_conditional_losses_1971952912%
#vae_encoder/StatefulPartitionedCall?
'twin_bottleneck/StatefulPartitionedCallStatefulPartitionedCall,vae_encoder/StatefulPartitionedCall:output:0,vae_encoder/StatefulPartitionedCall:output:1twin_bottleneck_197195700twin_bottleneck_197195702*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_1971954322)
'twin_bottleneck/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall,vae_encoder/StatefulPartitionedCall:output:0dense_6_197195705dense_6_197195707*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_6_layer_call_and_return_conditional_losses_1971954822!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall0twin_bottleneck/StatefulPartitionedCall:output:0dense_7_197195710dense_7_197195712*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_7_layer_call_and_return_conditional_losses_1971955092!
dense_7/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall0twin_bottleneck/StatefulPartitionedCall:output:0decoder_197195715decoder_197195717decoder_197195719decoder_197195721*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_decoder_layer_call_and_return_conditional_losses_1971955432!
decoder/StatefulPartitionedCall?
!dense_6/StatefulPartitionedCall_1StatefulPartitionedCallinputs_3dense_6_197195705dense_6_197195707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_6_layer_call_and_return_conditional_losses_1971956082#
!dense_6/StatefulPartitionedCall_1?
!dense_7/StatefulPartitionedCall_1StatefulPartitionedCallinputs_4dense_7_197195710dense_7_197195712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_7_layer_call_and_return_conditional_losses_1971956312#
!dense_7/StatefulPartitionedCall_1?
IdentityIdentity,vae_encoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dense_6/StatefulPartitionedCall_1 ^dense_7/StatefulPartitionedCall"^dense_7/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall$^vae_encoder/StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dense_6/StatefulPartitionedCall_1 ^dense_7/StatefulPartitionedCall"^dense_7/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall$^vae_encoder/StatefulPartitionedCall*
T0*
_output_shapes
:	? 2

Identity_1?

Identity_2Identity(dense_6/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dense_6/StatefulPartitionedCall_1 ^dense_7/StatefulPartitionedCall"^dense_7/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall$^vae_encoder/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identity(dense_7/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dense_6/StatefulPartitionedCall_1 ^dense_7/StatefulPartitionedCall"^dense_7/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall$^vae_encoder/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identity*dense_6/StatefulPartitionedCall_1:output:0 ^decoder/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dense_6/StatefulPartitionedCall_1 ^dense_7/StatefulPartitionedCall"^dense_7/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall$^vae_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity*dense_7/StatefulPartitionedCall_1:output:0 ^decoder/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dense_6/StatefulPartitionedCall_1 ^dense_7/StatefulPartitionedCall"^dense_7/StatefulPartitionedCall_1(^twin_bottleneck/StatefulPartitionedCall$^vae_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:?????????? :?????????
:?????????@:??????????::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dense_6/StatefulPartitionedCall_1!dense_6/StatefulPartitionedCall_12B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dense_7/StatefulPartitionedCall_1!dense_7/StatefulPartitionedCall_12R
'twin_bottleneck/StatefulPartitionedCall'twin_bottleneck/StatefulPartitionedCall2J
#vae_encoder/StatefulPartitionedCall#vae_encoder/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
B__inference_tbh_layer_call_and_return_conditional_losses_197196062
	input_1_1
	input_1_2
	input_1_3
input_2
input_34
0vae_encoder_dense_matmul_readvariableop_resource5
1vae_encoder_dense_biasadd_readvariableop_resource6
2vae_encoder_dense_1_matmul_readvariableop_resource7
3vae_encoder_dense_1_biasadd_readvariableop_resource6
2vae_encoder_dense_2_matmul_readvariableop_resource7
3vae_encoder_dense_2_biasadd_readvariableop_resource
identity??(vae_encoder/dense/BiasAdd/ReadVariableOp?*vae_encoder/dense/BiasAdd_1/ReadVariableOp?'vae_encoder/dense/MatMul/ReadVariableOp?)vae_encoder/dense/MatMul_1/ReadVariableOp?*vae_encoder/dense_1/BiasAdd/ReadVariableOp?)vae_encoder/dense_1/MatMul/ReadVariableOp?*vae_encoder/dense_2/BiasAdd/ReadVariableOp?)vae_encoder/dense_2/MatMul/ReadVariableOp_
vae_encoder/ShapeShape	input_1_2*
T0*
_output_shapes
:2
vae_encoder/Shape?
vae_encoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
vae_encoder/strided_slice/stack?
!vae_encoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!vae_encoder/strided_slice/stack_1?
!vae_encoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!vae_encoder/strided_slice/stack_2?
vae_encoder/strided_sliceStridedSlicevae_encoder/Shape:output:0(vae_encoder/strided_slice/stack:output:0*vae_encoder/strided_slice/stack_1:output:0*vae_encoder/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vae_encoder/strided_slice?
'vae_encoder/dense/MatMul/ReadVariableOpReadVariableOp0vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02)
'vae_encoder/dense/MatMul/ReadVariableOp?
vae_encoder/dense/MatMulMatMul	input_1_2/vae_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/MatMul?
(vae_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp1vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(vae_encoder/dense/BiasAdd/ReadVariableOp?
vae_encoder/dense/BiasAddBiasAdd"vae_encoder/dense/MatMul:product:00vae_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/BiasAdd?
vae_encoder/dense/ReluRelu"vae_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/Relu?
)vae_encoder/dense/MatMul_1/ReadVariableOpReadVariableOp0vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02+
)vae_encoder/dense/MatMul_1/ReadVariableOp?
vae_encoder/dense/MatMul_1MatMul	input_1_21vae_encoder/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/MatMul_1?
*vae_encoder/dense/BiasAdd_1/ReadVariableOpReadVariableOp1vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense/BiasAdd_1/ReadVariableOp?
vae_encoder/dense/BiasAdd_1BiasAdd$vae_encoder/dense/MatMul_1:product:02vae_encoder/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/BiasAdd_1?
vae_encoder/dense/Relu_1Relu$vae_encoder/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/Relu_1?
)vae_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)vae_encoder/dense_1/MatMul/ReadVariableOp?
vae_encoder/dense_1/MatMulMatMul&vae_encoder/dense/Relu_1:activations:01vae_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_1/MatMul?
*vae_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense_1/BiasAdd/ReadVariableOp?
vae_encoder/dense_1/BiasAddBiasAdd$vae_encoder/dense_1/MatMul:product:02vae_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_1/BiasAddh
vae_encoder/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder/Const|
vae_encoder/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder/split/split_dim?
vae_encoder/splitSplit$vae_encoder/split/split_dim:output:0$vae_encoder/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
vae_encoder/split?
vae_encoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
vae_encoder/Reshape/shape?
vae_encoder/ReshapeReshapevae_encoder/split:output:0"vae_encoder/Reshape/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Reshape?
vae_encoder/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
vae_encoder/Reshape_1/shape?
vae_encoder/Reshape_1Reshapevae_encoder/split:output:1$vae_encoder/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Reshape_1{
vae_encoder/zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    2
vae_encoder/zerosp
vae_encoder/NegNegvae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Negg
vae_encoder/ExpExpvae_encoder/Neg:y:0*
T0*
_output_shapes

:@2
vae_encoder/Expt
vae_encoder/Neg_1Negvae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_1m
vae_encoder/Exp_1Expvae_encoder/Neg_1:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_1k
vae_encoder/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add/y?
vae_encoder/addAddV2vae_encoder/Exp_1:y:0vae_encoder/add/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/addk
vae_encoder/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/pow/y?
vae_encoder/powPowvae_encoder/add:z:0vae_encoder/pow/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/pow?
vae_encoder/truedivRealDivvae_encoder/Exp:y:0vae_encoder/pow:z:0*
T0*
_output_shapes

:@2
vae_encoder/truedivv
vae_encoder/Neg_2Negvae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_2m
vae_encoder/Exp_2Expvae_encoder/Neg_2:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_2v
vae_encoder/Neg_3Negvae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_3m
vae_encoder/Exp_3Expvae_encoder/Neg_3:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_3o
vae_encoder/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add_1/y?
vae_encoder/add_1AddV2vae_encoder/Exp_3:y:0vae_encoder/add_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_1o
vae_encoder/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/pow_1/y?
vae_encoder/pow_1Powvae_encoder/add_1:z:0vae_encoder/pow_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/pow_1?
vae_encoder/truediv_1RealDivvae_encoder/Exp_2:y:0vae_encoder/pow_1:z:0*
T0*
_output_shapes

:@2
vae_encoder/truediv_1?
vae_encoder/mulMulvae_encoder/zeros:output:0vae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/mul?
vae_encoder/add_2AddV2vae_encoder/mul:z:0vae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_2l
vae_encoder/SignSignvae_encoder/add_2:z:0*
T0*
_output_shapes

:@2
vae_encoder/Signo
vae_encoder/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add_3/y?
vae_encoder/add_3AddV2vae_encoder/Sign:y:0vae_encoder/add_3/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_3w
vae_encoder/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/truediv_2/y?
vae_encoder/truediv_2RealDivvae_encoder/add_3:z:0 vae_encoder/truediv_2/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/truediv_2|
vae_encoder/IdentityIdentityvae_encoder/truediv_2:z:0*
T0*
_output_shapes

:@2
vae_encoder/Identity?
vae_encoder/IdentityN	IdentityNvae_encoder/truediv_2:z:0vae_encoder/Reshape:output:0vae_encoder/Reshape_1:output:0vae_encoder/zeros:output:0*
T
2*/
_gradient_op_typeCustomGradient-197196023*<
_output_shapes*
(:@:@:@:@2
vae_encoder/IdentityN?
)vae_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)vae_encoder/dense_2/MatMul/ReadVariableOp?
vae_encoder/dense_2/MatMulMatMul$vae_encoder/dense/Relu:activations:01vae_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/MatMul?
*vae_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense_2/BiasAdd/ReadVariableOp?
vae_encoder/dense_2/BiasAddBiasAdd$vae_encoder/dense_2/MatMul:product:02vae_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/BiasAdd?
vae_encoder/dense_2/SigmoidSigmoid$vae_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/Sigmoid?
IdentityIdentityvae_encoder/IdentityN:output:0)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:?????????? :?????????
:?????????@:??????????::::::2T
(vae_encoder/dense/BiasAdd/ReadVariableOp(vae_encoder/dense/BiasAdd/ReadVariableOp2X
*vae_encoder/dense/BiasAdd_1/ReadVariableOp*vae_encoder/dense/BiasAdd_1/ReadVariableOp2R
'vae_encoder/dense/MatMul/ReadVariableOp'vae_encoder/dense/MatMul/ReadVariableOp2V
)vae_encoder/dense/MatMul_1/ReadVariableOp)vae_encoder/dense/MatMul_1/ReadVariableOp2X
*vae_encoder/dense_1/BiasAdd/ReadVariableOp*vae_encoder/dense_1/BiasAdd/ReadVariableOp2V
)vae_encoder/dense_1/MatMul/ReadVariableOp)vae_encoder/dense_1/MatMul/ReadVariableOp2X
*vae_encoder/dense_2/BiasAdd/ReadVariableOp*vae_encoder/dense_2/BiasAdd/ReadVariableOp2V
)vae_encoder/dense_2/MatMul/ReadVariableOp)vae_encoder/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:?????????? 
#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????

#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
??
?
B__inference_tbh_layer_call_and_return_conditional_losses_197196271

inputs_0_0

inputs_0_1

inputs_0_2
inputs_1
inputs_24
0vae_encoder_dense_matmul_readvariableop_resource5
1vae_encoder_dense_biasadd_readvariableop_resource6
2vae_encoder_dense_1_matmul_readvariableop_resource7
3vae_encoder_dense_1_biasadd_readvariableop_resource6
2vae_encoder_dense_2_matmul_readvariableop_resource7
3vae_encoder_dense_2_biasadd_readvariableop_resource'
#twin_bottleneck_gcn_layer_197196221'
#twin_bottleneck_gcn_layer_197196223*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource2
.decoder_dense_3_matmul_readvariableop_resource3
/decoder_dense_3_biasadd_readvariableop_resource2
.decoder_dense_4_matmul_readvariableop_resource3
/decoder_dense_4_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5??&decoder/dense_3/BiasAdd/ReadVariableOp?%decoder/dense_3/MatMul/ReadVariableOp?&decoder/dense_4/BiasAdd/ReadVariableOp?%decoder/dense_4/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp? dense_6/BiasAdd_1/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_6/MatMul_1/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp? dense_7/BiasAdd_1/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_7/MatMul_1/ReadVariableOp?1twin_bottleneck/gcn_layer/StatefulPartitionedCall?(vae_encoder/dense/BiasAdd/ReadVariableOp?*vae_encoder/dense/BiasAdd_1/ReadVariableOp?'vae_encoder/dense/MatMul/ReadVariableOp?)vae_encoder/dense/MatMul_1/ReadVariableOp?*vae_encoder/dense_1/BiasAdd/ReadVariableOp?)vae_encoder/dense_1/MatMul/ReadVariableOp?*vae_encoder/dense_2/BiasAdd/ReadVariableOp?)vae_encoder/dense_2/MatMul/ReadVariableOp`
vae_encoder/ShapeShape
inputs_0_1*
T0*
_output_shapes
:2
vae_encoder/Shape?
vae_encoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
vae_encoder/strided_slice/stack?
!vae_encoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!vae_encoder/strided_slice/stack_1?
!vae_encoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!vae_encoder/strided_slice/stack_2?
vae_encoder/strided_sliceStridedSlicevae_encoder/Shape:output:0(vae_encoder/strided_slice/stack:output:0*vae_encoder/strided_slice/stack_1:output:0*vae_encoder/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vae_encoder/strided_slice?
'vae_encoder/dense/MatMul/ReadVariableOpReadVariableOp0vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02)
'vae_encoder/dense/MatMul/ReadVariableOp?
vae_encoder/dense/MatMulMatMul
inputs_0_1/vae_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/MatMul?
(vae_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp1vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(vae_encoder/dense/BiasAdd/ReadVariableOp?
vae_encoder/dense/BiasAddBiasAdd"vae_encoder/dense/MatMul:product:00vae_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/BiasAdd?
vae_encoder/dense/ReluRelu"vae_encoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/Relu?
)vae_encoder/dense/MatMul_1/ReadVariableOpReadVariableOp0vae_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
? ?*
dtype02+
)vae_encoder/dense/MatMul_1/ReadVariableOp?
vae_encoder/dense/MatMul_1MatMul
inputs_0_11vae_encoder/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/MatMul_1?
*vae_encoder/dense/BiasAdd_1/ReadVariableOpReadVariableOp1vae_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense/BiasAdd_1/ReadVariableOp?
vae_encoder/dense/BiasAdd_1BiasAdd$vae_encoder/dense/MatMul_1:product:02vae_encoder/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/BiasAdd_1?
vae_encoder/dense/Relu_1Relu$vae_encoder/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense/Relu_1?
)vae_encoder/dense_1/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)vae_encoder/dense_1/MatMul/ReadVariableOp?
vae_encoder/dense_1/MatMulMatMul&vae_encoder/dense/Relu_1:activations:01vae_encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_1/MatMul?
*vae_encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense_1/BiasAdd/ReadVariableOp?
vae_encoder/dense_1/BiasAddBiasAdd$vae_encoder/dense_1/MatMul:product:02vae_encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_1/BiasAddh
vae_encoder/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder/Const|
vae_encoder/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vae_encoder/split/split_dim?
vae_encoder/splitSplit$vae_encoder/split/split_dim:output:0$vae_encoder/dense_1/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????@:?????????@*
	num_split2
vae_encoder/split?
vae_encoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
vae_encoder/Reshape/shape?
vae_encoder/ReshapeReshapevae_encoder/split:output:0"vae_encoder/Reshape/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Reshape?
vae_encoder/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
vae_encoder/Reshape_1/shape?
vae_encoder/Reshape_1Reshapevae_encoder/split:output:1$vae_encoder/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Reshape_1?
vae_encoder/random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2!
vae_encoder/random_normal/shape?
vae_encoder/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
vae_encoder/random_normal/mean?
 vae_encoder/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 vae_encoder/random_normal/stddev?
.vae_encoder/random_normal/RandomStandardNormalRandomStandardNormal(vae_encoder/random_normal/shape:output:0*
T0*
_output_shapes

:@*
dtype020
.vae_encoder/random_normal/RandomStandardNormal?
vae_encoder/random_normal/mulMul7vae_encoder/random_normal/RandomStandardNormal:output:0)vae_encoder/random_normal/stddev:output:0*
T0*
_output_shapes

:@2
vae_encoder/random_normal/mul?
vae_encoder/random_normalAdd!vae_encoder/random_normal/mul:z:0'vae_encoder/random_normal/mean:output:0*
T0*
_output_shapes

:@2
vae_encoder/random_normal?
#vae_encoder/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#vae_encoder/clip_by_value/Minimum/y?
!vae_encoder/clip_by_value/MinimumMinimumvae_encoder/random_normal:z:0,vae_encoder/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:@2#
!vae_encoder/clip_by_value/Minimum
vae_encoder/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/clip_by_value/y?
vae_encoder/clip_by_valueMaximum%vae_encoder/clip_by_value/Minimum:z:0$vae_encoder/clip_by_value/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/clip_by_valuep
vae_encoder/NegNegvae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Negg
vae_encoder/ExpExpvae_encoder/Neg:y:0*
T0*
_output_shapes

:@2
vae_encoder/Expt
vae_encoder/Neg_1Negvae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_1m
vae_encoder/Exp_1Expvae_encoder/Neg_1:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_1k
vae_encoder/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add/y?
vae_encoder/addAddV2vae_encoder/Exp_1:y:0vae_encoder/add/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/addk
vae_encoder/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/pow/y?
vae_encoder/powPowvae_encoder/add:z:0vae_encoder/pow/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/pow?
vae_encoder/truedivRealDivvae_encoder/Exp:y:0vae_encoder/pow:z:0*
T0*
_output_shapes

:@2
vae_encoder/truedivv
vae_encoder/Neg_2Negvae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_2m
vae_encoder/Exp_2Expvae_encoder/Neg_2:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_2v
vae_encoder/Neg_3Negvae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/Neg_3m
vae_encoder/Exp_3Expvae_encoder/Neg_3:y:0*
T0*
_output_shapes

:@2
vae_encoder/Exp_3o
vae_encoder/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add_1/y?
vae_encoder/add_1AddV2vae_encoder/Exp_3:y:0vae_encoder/add_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_1o
vae_encoder/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/pow_1/y?
vae_encoder/pow_1Powvae_encoder/add_1:z:0vae_encoder/pow_1/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/pow_1?
vae_encoder/truediv_1RealDivvae_encoder/Exp_2:y:0vae_encoder/pow_1:z:0*
T0*
_output_shapes

:@2
vae_encoder/truediv_1?
vae_encoder/mulMulvae_encoder/clip_by_value:z:0vae_encoder/Reshape_1:output:0*
T0*
_output_shapes

:@2
vae_encoder/mul?
vae_encoder/add_2AddV2vae_encoder/mul:z:0vae_encoder/Reshape:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_2l
vae_encoder/SignSignvae_encoder/add_2:z:0*
T0*
_output_shapes

:@2
vae_encoder/Signo
vae_encoder/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
vae_encoder/add_3/y?
vae_encoder/add_3AddV2vae_encoder/Sign:y:0vae_encoder/add_3/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/add_3w
vae_encoder/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
vae_encoder/truediv_2/y?
vae_encoder/truediv_2RealDivvae_encoder/add_3:z:0 vae_encoder/truediv_2/y:output:0*
T0*
_output_shapes

:@2
vae_encoder/truediv_2|
vae_encoder/IdentityIdentityvae_encoder/truediv_2:z:0*
T0*
_output_shapes

:@2
vae_encoder/Identity?
vae_encoder/IdentityN	IdentityNvae_encoder/truediv_2:z:0vae_encoder/Reshape:output:0vae_encoder/Reshape_1:output:0vae_encoder/clip_by_value:z:0*
T
2*/
_gradient_op_typeCustomGradient-197196182*<
_output_shapes*
(:@:@:@:@2
vae_encoder/IdentityN?
)vae_encoder/dense_2/MatMul/ReadVariableOpReadVariableOp2vae_encoder_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)vae_encoder/dense_2/MatMul/ReadVariableOp?
vae_encoder/dense_2/MatMulMatMul$vae_encoder/dense/Relu:activations:01vae_encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/MatMul?
*vae_encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp3vae_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*vae_encoder/dense_2/BiasAdd/ReadVariableOp?
vae_encoder/dense_2/BiasAddBiasAdd$vae_encoder/dense_2/MatMul:product:02vae_encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/BiasAdd?
vae_encoder/dense_2/SigmoidSigmoid$vae_encoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
vae_encoder/dense_2/Sigmoid?
twin_bottleneck/PartitionedCallPartitionedCallvae_encoder/IdentityN:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_build_adjacency_hamming_142320222!
twin_bottleneck/PartitionedCall?
1twin_bottleneck/gcn_layer/StatefulPartitionedCallStatefulPartitionedCallvae_encoder/dense_2/Sigmoid:y:0(twin_bottleneck/PartitionedCall:output:0#twin_bottleneck_gcn_layer_197196221#twin_bottleneck_gcn_layer_197196223*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_spectrum_conv_1423207323
1twin_bottleneck/gcn_layer/StatefulPartitionedCall?
twin_bottleneck/SigmoidSigmoid:twin_bottleneck/gcn_layer/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	?2
twin_bottleneck/Sigmoid?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulvae_encoder/IdentityN:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_6/BiasAddp
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_6/Sigmoid?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMultwin_bottleneck/Sigmoid:y:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_7/BiasAddp
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_7/Sigmoid?
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%decoder/dense_3/MatMul/ReadVariableOp?
decoder/dense_3/MatMulMatMultwin_bottleneck/Sigmoid:y:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
decoder/dense_3/MatMul?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_3/BiasAdd/ReadVariableOp?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
decoder/dense_3/BiasAdd?
decoder/dense_3/ReluRelu decoder/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:	?2
decoder/dense_3/Relu?
%decoder/dense_4/MatMul/ReadVariableOpReadVariableOp.decoder_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02'
%decoder/dense_4/MatMul/ReadVariableOp?
decoder/dense_4/MatMulMatMul"decoder/dense_3/Relu:activations:0-decoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
decoder/dense_4/MatMul?
&decoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02(
&decoder/dense_4/BiasAdd/ReadVariableOp?
decoder/dense_4/BiasAddBiasAdd decoder/dense_4/MatMul:product:0.decoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
decoder/dense_4/BiasAdd?
decoder/dense_4/ReluRelu decoder/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:	? 2
decoder/dense_4/Relu?
dense_6/MatMul_1/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_6/MatMul_1/ReadVariableOp?
dense_6/MatMul_1MatMulinputs_1'dense_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul_1?
 dense_6/BiasAdd_1/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_6/BiasAdd_1/ReadVariableOp?
dense_6/BiasAdd_1BiasAdddense_6/MatMul_1:product:0(dense_6/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAdd_1
dense_6/Sigmoid_1Sigmoiddense_6/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
dense_6/Sigmoid_1?
dense_7/MatMul_1/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_7/MatMul_1/ReadVariableOp?
dense_7/MatMul_1MatMulinputs_2'dense_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul_1?
 dense_7/BiasAdd_1/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_7/BiasAdd_1/ReadVariableOp?
dense_7/BiasAdd_1BiasAdddense_7/MatMul_1:product:0(dense_7/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAdd_1
dense_7/Sigmoid_1Sigmoiddense_7/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
dense_7/Sigmoid_1?
IdentityIdentityvae_encoder/IdentityN:output:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:@2

Identity?

Identity_1Identity"decoder/dense_4/Relu:activations:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	? 2

Identity_1?

Identity_2Identitydense_6/Sigmoid:y:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_2?

Identity_3Identitydense_7/Sigmoid:y:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity_3?

Identity_4Identitydense_6/Sigmoid_1:y:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_7/Sigmoid_1:y:0'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp'^decoder/dense_4/BiasAdd/ReadVariableOp&^decoder/dense_4/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/BiasAdd_1/ReadVariableOp^dense_6/MatMul/ReadVariableOp ^dense_6/MatMul_1/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/BiasAdd_1/ReadVariableOp^dense_7/MatMul/ReadVariableOp ^dense_7/MatMul_1/ReadVariableOp2^twin_bottleneck/gcn_layer/StatefulPartitionedCall)^vae_encoder/dense/BiasAdd/ReadVariableOp+^vae_encoder/dense/BiasAdd_1/ReadVariableOp(^vae_encoder/dense/MatMul/ReadVariableOp*^vae_encoder/dense/MatMul_1/ReadVariableOp+^vae_encoder/dense_1/BiasAdd/ReadVariableOp*^vae_encoder/dense_1/MatMul/ReadVariableOp+^vae_encoder/dense_2/BiasAdd/ReadVariableOp*^vae_encoder/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes?
?:?????????:?????????? :?????????
:?????????@:??????????::::::::::::::::2P
&decoder/dense_3/BiasAdd/ReadVariableOp&decoder/dense_3/BiasAdd/ReadVariableOp2N
%decoder/dense_3/MatMul/ReadVariableOp%decoder/dense_3/MatMul/ReadVariableOp2P
&decoder/dense_4/BiasAdd/ReadVariableOp&decoder/dense_4/BiasAdd/ReadVariableOp2N
%decoder/dense_4/MatMul/ReadVariableOp%decoder/dense_4/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/BiasAdd_1/ReadVariableOp dense_6/BiasAdd_1/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2B
dense_6/MatMul_1/ReadVariableOpdense_6/MatMul_1/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/BiasAdd_1/ReadVariableOp dense_7/BiasAdd_1/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2B
dense_7/MatMul_1/ReadVariableOpdense_7/MatMul_1/ReadVariableOp2f
1twin_bottleneck/gcn_layer/StatefulPartitionedCall1twin_bottleneck/gcn_layer/StatefulPartitionedCall2T
(vae_encoder/dense/BiasAdd/ReadVariableOp(vae_encoder/dense/BiasAdd/ReadVariableOp2X
*vae_encoder/dense/BiasAdd_1/ReadVariableOp*vae_encoder/dense/BiasAdd_1/ReadVariableOp2R
'vae_encoder/dense/MatMul/ReadVariableOp'vae_encoder/dense/MatMul/ReadVariableOp2V
)vae_encoder/dense/MatMul_1/ReadVariableOp)vae_encoder/dense/MatMul_1/ReadVariableOp2X
*vae_encoder/dense_1/BiasAdd/ReadVariableOp*vae_encoder/dense_1/BiasAdd/ReadVariableOp2V
)vae_encoder/dense_1/MatMul/ReadVariableOp)vae_encoder/dense_1/MatMul/ReadVariableOp2X
*vae_encoder/dense_2/BiasAdd/ReadVariableOp*vae_encoder/dense_2/BiasAdd/ReadVariableOp2V
)vae_encoder/dense_2/MatMul/ReadVariableOp)vae_encoder/dense_2/MatMul/ReadVariableOp:O K
#
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:TP
(
_output_shapes
:?????????? 
$
_user_specified_name
inputs/0/1:SO
'
_output_shapes
:?????????

$
_user_specified_name
inputs/0/2:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
?
+__inference_dense_7_layer_call_fn_197196784

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_7_layer_call_and_return_conditional_losses_1971956312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_6_layer_call_fn_197196744

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_6_layer_call_and_return_conditional_losses_1971954822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:@
 
_user_specified_nameinputs
?
?
F__inference_decoder_layer_call_and_return_conditional_losses_197195543

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_3/BiasAddh
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*
_output_shapes
:	?2
dense_3/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_4/BiasAddh
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*
_output_shapes
:	? 2
dense_4/Relu?
IdentityIdentitydense_4/Relu:activations:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*
_output_shapes
:	? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?	
?
F__inference_dense_6_layer_call_and_return_conditional_losses_197195482

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SigmoidSigmoidBiasAdd:output:0*
T0*
_output_shapes

:2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:@
 
_user_specified_nameinputs
?
K
,__inference_build_adjacency_hamming_14232022
	tensor_in
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
CastS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/yU
subSub	tensor_insub/y:output:0*
T0*
_output_shapes

:@2
subj
MatMulMatMul	tensor_insub:z:0*
T0*
_output_shapes

:*
transpose_b(2
MatMuln
MatMul_1MatMulsub:z:0	tensor_in*
T0*
_output_shapes

:*
transpose_b(2

MatMul_1b
addAddV2MatMul:product:0MatMul_1:product:0*
T0*
_output_shapes

:2
addC
AbsAbsadd:z:0*
T0*
_output_shapes

:2
AbsY
truedivRealDivAbs:y:0Cast:y:0*
T0*
_output_shapes

:2	
truedivW
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
sub_1/x]
sub_1Subsub_1/x:output:0truediv:z:0*
T0*
_output_shapes

:2
sub_1S
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *33??2
Pow/yU
PowPow	sub_1:z:0Pow/y:output:0*
T0*
_output_shapes

:2
PowR
IdentityIdentityPow:z:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

:@:I E

_output_shapes

:@
#
_user_specified_name	tensor_in
?
?
"__inference_spectrum_conv_14233590

values
	adjacency*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulvalues%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAdd?
PartitionedCallPartitionedCall	adjacency*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_graph_laplacian_142320692
PartitionedCallx
matmulMatMulPartitionedCall:output:0dense_5/BiasAdd:output:0*
T0*
_output_shapes
:	?2
matmul?
IdentityIdentitymatmul:product:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:??????????:::2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_namevalues:IE

_output_shapes

:
#
_user_specified_name	adjacency
?
?
'__inference_signature_wrapper_197195847
	input_1_1
	input_1_2
	input_1_3
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_1_1	input_1_2	input_1_3input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__wrapped_model_1971951992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:?????????? :?????????
:?????????@:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:?????????? 
#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????

#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
3__inference_twin_bottleneck_layer_call_fn_197196714
bbn
cbn
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbbncbnunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_1971954442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:@:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:C ?

_output_shapes

:@

_user_specified_namebbn:MI
(
_output_shapes
:??????????

_user_specified_namecbn
?
C
$__inference_graph_laplacian_14233527
	adjacency
identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceZ

ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2

ones/mul/yi
ones/mulMulstrided_slice:output:0ones/mul/y:output:0*
T0*
_output_shapes
: 2

ones/mul]
ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
ones/Less/yc
	ones/LessLessones/mul:z:0ones/Less/y:output:0*
T0*
_output_shapes
: 2
	ones/Less`
ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/packed/1?
ones/packedPackstrided_slice:output:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:2
ones/packed]

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

ones/Consth
onesFillones/packed:output:0ones/Const:output:0*
T0*
_output_shapes

:2
ones]
matmulMatMul	adjacencyones:output:0*
T0*
_output_shapes

:2
matmulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+22
add/y^
addAddV2matmul:product:0add/y:output:0*
T0*
_output_shapes

:2
addS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Pow/yS
PowPowadd:z:0Pow/y:output:0*
T0*
_output_shapes

:2
Powv
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
eye/MinimumY
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
	eye/shapeq
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:2
eye/concat/values_1d
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
eye/concat/axis?

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:2

eye/concate
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
eye/ones/Consto
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes
:2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_value?
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diagV
mulMuleye/diag:output:0Pow:z:0*
T0*
_output_shapes

:2
mul[
matmul_1MatMulmul:z:0	adjacency*
T0*
_output_shapes

:2

matmul_1d
matmul_2MatMulmatmul_1:product:0mul:z:0*
T0*
_output_shapes

:2

matmul_2]
IdentityIdentitymatmul_2:product:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

::I E

_output_shapes

:
#
_user_specified_name	adjacency
?
?
'__inference_tbh_layer_call_fn_197196134
	input_1_1
	input_1_2
	input_1_3
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_1_1	input_1_2	input_1_3input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_tbh_layer_call_and_return_conditional_losses_1971958092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:@2

Identity"
identityIdentity:output:0*?
_input_shapesw
u:?????????:?????????? :?????????
:?????????@:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:?????????
#
_user_specified_name	input_1_1:SO
(
_output_shapes
:?????????? 
#
_user_specified_name	input_1_2:RN
'
_output_shapes
:?????????

#
_user_specified_name	input_1_3:PL
'
_output_shapes
:?????????@
!
_user_specified_name	input_2:QM
(
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
F__inference_decoder_layer_call_and_return_conditional_losses_197196654

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_3/BiasAddh
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*
_output_shapes
:	?2
dense_3/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	? 2
dense_4/BiasAddh
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*
_output_shapes
:	? 2
dense_4/Relu?
IdentityIdentitydense_4/Relu:activations:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*
_output_shapes
:	? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:	?::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
	input_1_1.
serving_default_input_1_1:0?????????
@
	input_1_23
serving_default_input_1_2:0?????????? 
?
	input_1_32
serving_default_input_1_3:0?????????

;
input_20
serving_default_input_2:0?????????@
<
input_31
serving_default_input_3:0??????????3
output_1'
StatefulPartitionedCall:0@tensorflow/serving/predict:??
?
encoder
decoder
tbn
	dis_1
	dis_2
	variables
trainable_variables
regularization_losses
		keras_api


signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?
_tf_keras_model?{"class_name": "TBH", "name": "tbh", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "TBH"}}
?
fc_1

fc_2_1

fc_2_2
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "VaeEncoder", "name": "vae_encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
fc_1
fc_2
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Decoder", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
gcn
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TwinBottleneck", "name": "twin_bottleneck", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 64]}}
?

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 512]}}
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
12
13
#14
$15"
trackable_list_wrapper
?
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
12
13
#14
$15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
5layer_metrics
regularization_losses

6layers
7layer_regularization_losses
8metrics
9non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

)kernel
*bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 4096]}}
?

+kernel
,bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 1024]}}
?

-kernel
.bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 1024]}}
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
Flayer_metrics
regularization_losses

Glayers
Hlayer_regularization_losses
Imetrics
Jnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

/kernel
0bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 512]}}
?

1kernel
2bias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 4096, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 1024]}}
<
/0
01
12
23"
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
Slayer_metrics
regularization_losses

Tlayers
Ulayer_regularization_losses
Vmetrics
Wnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Xfc
Yrs
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?graph_laplacian
?spectrum_conv
?spectrum_conv_adapt"?
_tf_keras_layer?{"class_name": "GCNLayer", "name": "gcn_layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
^layer_metrics
regularization_losses

_layers
`layer_regularization_losses
ametrics
bnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"@2tbh/dense_6/kernel
:2tbh/dense_6/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
 trainable_variables
clayer_metrics
!regularization_losses

dlayers
elayer_regularization_losses
fmetrics
gnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	?2tbh/dense_7/kernel
:2tbh/dense_7/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
%	variables
&trainable_variables
hlayer_metrics
'regularization_losses

ilayers
jlayer_regularization_losses
kmetrics
lnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.
? ?2tbh/vae_encoder/dense/kernel
):'?2tbh/vae_encoder/dense/bias
2:0
??2tbh/vae_encoder/dense_1/kernel
+:)?2tbh/vae_encoder/dense_1/bias
2:0
??2tbh/vae_encoder/dense_2/kernel
+:)?2tbh/vae_encoder/dense_2/bias
.:,
??2tbh/decoder/dense_3/kernel
':%?2tbh/decoder/dense_3/bias
.:,
?? 2tbh/decoder/dense_4/kernel
':%? 2tbh/decoder/dense_4/bias
": 
??2dense_5/kernel
:?2dense_5/bias
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
:	variables
;trainable_variables
mlayer_metrics
<regularization_losses

nlayers
olayer_regularization_losses
pmetrics
qnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>	variables
?trainable_variables
rlayer_metrics
@regularization_losses

slayers
tlayer_regularization_losses
umetrics
vnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
B	variables
Ctrainable_variables
wlayer_metrics
Dregularization_losses

xlayers
ylayer_regularization_losses
zmetrics
{non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
K	variables
Ltrainable_variables
|layer_metrics
Mregularization_losses

}layers
~layer_regularization_losses
metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
O	variables
Ptrainable_variables
?layer_metrics
Qregularization_losses
?layers
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

3kernel
4bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 512]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Z	variables
[trainable_variables
?layer_metrics
\regularization_losses
?layers
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
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
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_dict_wrapper
.
X0
Y1"
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
?2?
'__inference_tbh_layer_call_fn_197196113
'__inference_tbh_layer_call_fn_197196134
'__inference_tbh_layer_call_fn_197196421
'__inference_tbh_layer_call_fn_197196400?
???
FullArgSpec?
args7?4
jself
jinputs

jtraining
jmask
j
continuous
varargs
 
varkw
 
defaults?
p

 
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_tbh_layer_call_and_return_conditional_losses_197195984
B__inference_tbh_layer_call_and_return_conditional_losses_197196349
B__inference_tbh_layer_call_and_return_conditional_losses_197196271
B__inference_tbh_layer_call_and_return_conditional_losses_197196062?
???
FullArgSpec?
args7?4
jself
jinputs

jtraining
jmask
j
continuous
varargs
 
varkw
 
defaults?
p

 
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference__wrapped_model_197195199?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2?????????? 
#? 
	input_1_3?????????

!?
input_2?????????@
"?
input_3??????????
?2?
/__inference_vae_encoder_layer_call_fn_197196599
/__inference_vae_encoder_layer_call_fn_197196618?
???
FullArgSpec7
args/?,
jself
jinputs

jtraining
j
continuous
varargs
 
varkwjkwargs
defaults?
p
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_vae_encoder_layer_call_and_return_conditional_losses_197196580
J__inference_vae_encoder_layer_call_and_return_conditional_losses_197196505?
???
FullArgSpec7
args/?,
jself
jinputs

jtraining
j
continuous
varargs
 
varkwjkwargs
defaults?
p
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_decoder_layer_call_fn_197196667
+__inference_decoder_layer_call_fn_197196680?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
F__inference_decoder_layer_call_and_return_conditional_losses_197196654
F__inference_decoder_layer_call_and_return_conditional_losses_197196636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
3__inference_twin_bottleneck_layer_call_fn_197196714
3__inference_twin_bottleneck_layer_call_fn_197196724?
???
FullArgSpec-
args%?"
jself
jbbn
jcbn

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_197196692
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_197196704?
???
FullArgSpec-
args%?"
jself
jbbn
jcbn

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_6_layer_call_fn_197196744
+__inference_dense_6_layer_call_fn_197196764?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_6_layer_call_and_return_conditional_losses_197196735
F__inference_dense_6_layer_call_and_return_conditional_losses_197196755?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_7_layer_call_fn_197196784
+__inference_dense_7_layer_call_fn_197196804?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_7_layer_call_and_return_conditional_losses_197196795
F__inference_dense_7_layer_call_and_return_conditional_losses_197196775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_197195847	input_1_1	input_1_2	input_1_3input_2input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec*
args"?
jself
jvalues
j	adjacency
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec*
args"?
jself
jvalues
j	adjacency
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
$__inference_graph_laplacian_14233490
$__inference_graph_laplacian_14233527?
???
FullArgSpec
args?
j	adjacency
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference_spectrum_conv_14233577
"__inference_spectrum_conv_14233590?
???
FullArgSpec*
args"?
jself
jvalues
j	adjacency
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec*
args"?
jself
jvalues
j	adjacency
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
$__inference__wrapped_model_197195199?)*+,-.???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2?????????? 
#? 
	input_1_3?????????

!?
input_2?????????@
"?
input_3??????????
? "*?'
%
output_1?
output_1@?
F__inference_decoder_layer_call_and_return_conditional_losses_197196636^/0127?4
?
?
inputs	?
?

trainingp"?
?
0	? 
? ?
F__inference_decoder_layer_call_and_return_conditional_losses_197196654^/0127?4
?
?
inputs	?
?

trainingp "?
?
0	? 
? ?
+__inference_decoder_layer_call_fn_197196667Q/0127?4
?
?
inputs	?
?

trainingp"?	? ?
+__inference_decoder_layer_call_fn_197196680Q/0127?4
?
?
inputs	?
?

trainingp "?	? ?
F__inference_dense_6_layer_call_and_return_conditional_losses_197196735J&?#
?
?
inputs@
? "?
?
0
? ?
F__inference_dense_6_layer_call_and_return_conditional_losses_197196755\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? l
+__inference_dense_6_layer_call_fn_197196744=&?#
?
?
inputs@
? "?~
+__inference_dense_6_layer_call_fn_197196764O/?,
%?"
 ?
inputs?????????@
? "???????????
F__inference_dense_7_layer_call_and_return_conditional_losses_197196775]#$0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
F__inference_dense_7_layer_call_and_return_conditional_losses_197196795K#$'?$
?
?
inputs	?
? "?
?
0
? 
+__inference_dense_7_layer_call_fn_197196784P#$0?-
&?#
!?
inputs??????????
? "??????????m
+__inference_dense_7_layer_call_fn_197196804>#$'?$
?
?
inputs	?
? "?h
$__inference_graph_laplacian_14233490@+?(
!?
?
	adjacency
??
? "?
??d
$__inference_graph_laplacian_14233527<)?&
?
?
	adjacency
? "??
'__inference_signature_wrapper_197195847?)*+,-.???
? 
???
,
	input_1_1?
	input_1_1?????????
1
	input_1_2$?!
	input_1_2?????????? 
0
	input_1_3#? 
	input_1_3?????????

,
input_2!?
input_2?????????@
-
input_3"?
input_3??????????"*?'
%
output_1?
output_1@?
"__inference_spectrum_conv_14233577_34F?C
<?9
?
values
??
?
	adjacency
??
? "?
???
"__inference_spectrum_conv_14233590d34L?I
B??
!?
values??????????
?
	adjacency
? "?	??
B__inference_tbh_layer_call_and_return_conditional_losses_197195984?)*+,-.34#$/012???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2?????????? 
#? 
	input_1_3?????????

!?
input_2?????????@
"?
input_3??????????
p

 
p
? "???
???
?
0/0@
?
0/1	? 
?
0/2
?
0/3
?
0/4?????????
?
0/5?????????
? ?
B__inference_tbh_layer_call_and_return_conditional_losses_197196062?)*+,-.???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2?????????? 
#? 
	input_1_3?????????

!?
input_2?????????@
"?
input_3??????????
p 

 
p
? "?
?
0@
? ?
B__inference_tbh_layer_call_and_return_conditional_losses_197196271?)*+,-.34#$/012???
???
???
r?o
 ?

inputs/0/0?????????
%?"

inputs/0/1?????????? 
$?!

inputs/0/2?????????

"?
inputs/1?????????@
#? 
inputs/2??????????
p

 
p
? "???
???
?
0/0@
?
0/1	? 
?
0/2
?
0/3
?
0/4?????????
?
0/5?????????
? ?
B__inference_tbh_layer_call_and_return_conditional_losses_197196349?)*+,-.???
???
???
r?o
 ?

inputs/0/0?????????
%?"

inputs/0/1?????????? 
$?!

inputs/0/2?????????

"?
inputs/1?????????@
#? 
inputs/2??????????
p 

 
p
? "?
?
0@
? ?
'__inference_tbh_layer_call_fn_197196113?)*+,-.34#$/012???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2?????????? 
#? 
	input_1_3?????????

!?
input_2?????????@
"?
input_3??????????
p

 
p
? "???
?
0@
?
1	? 
?
2
?
3
?
4?????????
?
5??????????
'__inference_tbh_layer_call_fn_197196134?)*+,-.???
???
???
o?l
?
	input_1_1?????????
$?!
	input_1_2?????????? 
#? 
	input_1_3?????????

!?
input_2?????????@
"?
input_3??????????
p 

 
p
? "?@?
'__inference_tbh_layer_call_fn_197196400?)*+,-.34#$/012???
???
???
r?o
 ?

inputs/0/0?????????
%?"

inputs/0/1?????????? 
$?!

inputs/0/2?????????

"?
inputs/1?????????@
#? 
inputs/2??????????
p

 
p
? "???
?
0@
?
1	? 
?
2
?
3
?
4?????????
?
5??????????
'__inference_tbh_layer_call_fn_197196421?)*+,-.???
???
???
r?o
 ?

inputs/0/0?????????
%?"

inputs/0/1?????????? 
$?!

inputs/0/2?????????

"?
inputs/1?????????@
#? 
inputs/2??????????
p 

 
p
? "?@?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_197196692l34G?D
=?:
?
bbn@
?
cbn??????????
p
? "?
?
0	?
? ?
N__inference_twin_bottleneck_layer_call_and_return_conditional_losses_197196704l34G?D
=?:
?
bbn@
?
cbn??????????
p 
? "?
?
0	?
? ?
3__inference_twin_bottleneck_layer_call_fn_197196714_34G?D
=?:
?
bbn@
?
cbn??????????
p
? "?	??
3__inference_twin_bottleneck_layer_call_fn_197196724_34G?D
=?:
?
bbn@
?
cbn??????????
p 
? "?	??
J__inference_vae_encoder_layer_call_and_return_conditional_losses_197196505?)*+,-.8?5
.?+
!?
inputs?????????? 
p
p
? "C?@
9?6
?
0/0@
?
0/1??????????
? ?
J__inference_vae_encoder_layer_call_and_return_conditional_losses_197196580?)*+,-.8?5
.?+
!?
inputs?????????? 
p 
p
? "C?@
9?6
?
0/0@
?
0/1??????????
? ?
/__inference_vae_encoder_layer_call_fn_197196599y)*+,-.8?5
.?+
!?
inputs?????????? 
p
p
? "5?2
?
0@
?
1???????????
/__inference_vae_encoder_layer_call_fn_197196618y)*+,-.8?5
.?+
!?
inputs?????????? 
p 
p
? "5?2
?
0@
?
1??????????