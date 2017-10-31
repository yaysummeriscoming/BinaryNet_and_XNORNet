{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fnil\fcharset0 Consolas;\f1\fswiss\fcharset0 Helvetica;\f2\fnil\fcharset0 LucidaGrande-Bold;
}
{\colortbl;\red255\green255\blue255;\red27\green31\blue34;\red244\green246\blue249;\red0\green0\blue0;
\red255\green255\blue255;\red27\green31\blue34;\red244\green246\blue249;}
{\*\expandedcolortbl;;\cssrgb\c14118\c16078\c18039;\cssrgb\c96471\c97255\c98039;\cssrgb\c0\c0\c0;
\cssrgb\c100000\c100000\c100000;\cssrgb\c14118\c16078\c18039;\cssrgb\c96471\c97255\c98039;}
\paperw11900\paperh16840\margl1440\margr1440\vieww19600\viewh13800\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs26 \cf0 This project contains Keras implementations of the BinaryNet and XNORNet papers:
\f1\fs24 \
\
\pard\pardeftab720\sl380\partightenfactor0

\f0\fs27\fsmilli13600 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 [
\f2\b\fs36 \cf4 \cb5 \strokec4 Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1
\f0\b0\fs27\fsmilli13600 \cf2 \cb3 \strokec2 ]({\field{\*\fldinst{HYPERLINK "https://arxiv.org/abs/1602.02830"}}{\fldrslt https://arxiv.org/abs/1602.02830}})\
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf6 \cb7 \outl0\strokewidth0 [
\f2\b\fs36 \cf4 \cb5 \outl0\strokewidth0 \strokec4 XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
\f0\b0\fs27\fsmilli13600 \cf6 \cb7 \outl0\strokewidth0 ]({\field{\*\fldinst{HYPERLINK "https://arxiv.org/abs/1602.02830"}}{\fldrslt https://arxiv.org/abs/1603.05279}})\
\
Code supports the Tensorflow and Theano backends.\
\
The most difficult part of coding these implementations was the sign function gradient.  I\'92ve used the clipped \'91passthrough\'92 sign implementation detailed in the BinaryNet paper.  The XNORNet doesn\'92t mention anything, so I\'92ve used the same implementation here too.}