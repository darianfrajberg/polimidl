#include <polimidl/network.hpp>
#include <polimidl/layers/batch_norm_relu.hpp>
#include <polimidl/layers/convolution.hpp>
#include <polimidl/layers/avg_pooling.hpp>
#include <polimidl/layers/depthwise_convolution.hpp>
#include <polimidl/layers/pointwise_convolution.hpp>
#include <polimidl/layers/softmax.hpp>
#include <polimidl/layers/bias.hpp>

#include "./mobilenet_weights/1convolution.h"
#include "./mobilenet_weights/2bnbeta.h"
#include "./mobilenet_weights/2bngamma_variance_epsilon.h"
#include "./mobilenet_weights/2bnmean.h"
#include "./mobilenet_weights/3depthwise_convolution.h"
#include "./mobilenet_weights/4bnbeta.h"
#include "./mobilenet_weights/4bngamma_variance_epsilon.h"
#include "./mobilenet_weights/4bnmean.h"
#include "./mobilenet_weights/5pointwise_convolution.h"
#include "./mobilenet_weights/6bnbeta.h"
#include "./mobilenet_weights/6bngamma_variance_epsilon.h"
#include "./mobilenet_weights/6bnmean.h"
#include "./mobilenet_weights/7depthwise_convolution.h"
#include "./mobilenet_weights/8bnbeta.h"
#include "./mobilenet_weights/8bngamma_variance_epsilon.h"
#include "./mobilenet_weights/8bnmean.h"
#include "./mobilenet_weights/9pointwise_convolution.h"
#include "./mobilenet_weights/10bnbeta.h"
#include "./mobilenet_weights/10bngamma_variance_epsilon.h"
#include "./mobilenet_weights/10bnmean.h"
#include "./mobilenet_weights/11depthwise_convolution.h"
#include "./mobilenet_weights/12bnbeta.h"
#include "./mobilenet_weights/12bngamma_variance_epsilon.h"
#include "./mobilenet_weights/12bnmean.h"
#include "./mobilenet_weights/13pointwise_convolution.h"
#include "./mobilenet_weights/14bnbeta.h"
#include "./mobilenet_weights/14bngamma_variance_epsilon.h"
#include "./mobilenet_weights/14bnmean.h"
#include "./mobilenet_weights/15depthwise_convolution.h"
#include "./mobilenet_weights/16bnbeta.h"
#include "./mobilenet_weights/16bngamma_variance_epsilon.h"
#include "./mobilenet_weights/16bnmean.h"
#include "./mobilenet_weights/17pointwise_convolution.h"
#include "./mobilenet_weights/18bnbeta.h"
#include "./mobilenet_weights/18bngamma_variance_epsilon.h"
#include "./mobilenet_weights/18bnmean.h"
#include "./mobilenet_weights/19depthwise_convolution.h"
#include "./mobilenet_weights/20bnbeta.h"
#include "./mobilenet_weights/20bngamma_variance_epsilon.h"
#include "./mobilenet_weights/20bnmean.h"
#include "./mobilenet_weights/21pointwise_convolution.h"
#include "./mobilenet_weights/22bnbeta.h"
#include "./mobilenet_weights/22bngamma_variance_epsilon.h"
#include "./mobilenet_weights/22bnmean.h"
#include "./mobilenet_weights/23depthwise_convolution.h"
#include "./mobilenet_weights/24bnbeta.h"
#include "./mobilenet_weights/24bngamma_variance_epsilon.h"
#include "./mobilenet_weights/24bnmean.h"
#include "./mobilenet_weights/25pointwise_convolution.h"
#include "./mobilenet_weights/26bnbeta.h"
#include "./mobilenet_weights/26bngamma_variance_epsilon.h"
#include "./mobilenet_weights/26bnmean.h"
#include "./mobilenet_weights/27depthwise_convolution.h"
#include "./mobilenet_weights/28bnbeta.h"
#include "./mobilenet_weights/28bngamma_variance_epsilon.h"
#include "./mobilenet_weights/28bnmean.h"
#include "./mobilenet_weights/29pointwise_convolution.h"
#include "./mobilenet_weights/30bnbeta.h"
#include "./mobilenet_weights/30bngamma_variance_epsilon.h"
#include "./mobilenet_weights/30bnmean.h"
#include "./mobilenet_weights/31depthwise_convolution.h"
#include "./mobilenet_weights/32bnbeta.h"
#include "./mobilenet_weights/32bngamma_variance_epsilon.h"
#include "./mobilenet_weights/32bnmean.h"
#include "./mobilenet_weights/33pointwise_convolution.h"
#include "./mobilenet_weights/34bnbeta.h"
#include "./mobilenet_weights/34bngamma_variance_epsilon.h"
#include "./mobilenet_weights/34bnmean.h"
#include "./mobilenet_weights/35depthwise_convolution.h"
#include "./mobilenet_weights/36bnbeta.h"
#include "./mobilenet_weights/36bngamma_variance_epsilon.h"
#include "./mobilenet_weights/36bnmean.h"
#include "./mobilenet_weights/37pointwise_convolution.h"
#include "./mobilenet_weights/38bnbeta.h"
#include "./mobilenet_weights/38bngamma_variance_epsilon.h"
#include "./mobilenet_weights/38bnmean.h"
#include "./mobilenet_weights/39depthwise_convolution.h"
#include "./mobilenet_weights/40bnbeta.h"
#include "./mobilenet_weights/40bngamma_variance_epsilon.h"
#include "./mobilenet_weights/40bnmean.h"
#include "./mobilenet_weights/41pointwise_convolution.h"
#include "./mobilenet_weights/42bnbeta.h"
#include "./mobilenet_weights/42bngamma_variance_epsilon.h"
#include "./mobilenet_weights/42bnmean.h"
#include "./mobilenet_weights/43depthwise_convolution.h"
#include "./mobilenet_weights/44bnbeta.h"
#include "./mobilenet_weights/44bngamma_variance_epsilon.h"
#include "./mobilenet_weights/44bnmean.h"
#include "./mobilenet_weights/45pointwise_convolution.h"
#include "./mobilenet_weights/46bnbeta.h"
#include "./mobilenet_weights/46bngamma_variance_epsilon.h"
#include "./mobilenet_weights/46bnmean.h"
#include "./mobilenet_weights/47depthwise_convolution.h"
#include "./mobilenet_weights/48bnbeta.h"
#include "./mobilenet_weights/48bngamma_variance_epsilon.h"
#include "./mobilenet_weights/48bnmean.h"
#include "./mobilenet_weights/49pointwise_convolution.h"
#include "./mobilenet_weights/50bnbeta.h"
#include "./mobilenet_weights/50bngamma_variance_epsilon.h"
#include "./mobilenet_weights/50bnmean.h"
#include "./mobilenet_weights/51depthwise_convolution.h"
#include "./mobilenet_weights/52bnbeta.h"
#include "./mobilenet_weights/52bngamma_variance_epsilon.h"
#include "./mobilenet_weights/52bnmean.h"
#include "./mobilenet_weights/53pointwise_convolution.h"
#include "./mobilenet_weights/54bnbeta.h"
#include "./mobilenet_weights/54bngamma_variance_epsilon.h"
#include "./mobilenet_weights/54bnmean.h"
#include "./mobilenet_weights/55pointwise_convolution.h"
#include "./mobilenet_weights/56bias.h"

auto build_mobilenet(std::size_t h, std::size_t w, unsigned int number_of_workers) {
	using polimidl::build_network;
	using polimidl::layers::components;
	using polimidl::layers::kernel;
	using polimidl::layers::padding;
	using polimidl::layers::stride;
	using polimidl::layers::batch_norm_relu;
	using polimidl::layers::convolution;
	using polimidl::layers::avg_pooling;
	using polimidl::layers::depthwise_convolution;
	using polimidl::layers::pointwise_convolution;
	using polimidl::layers::softmax;
	using polimidl::layers::bias;

	return build_network<float>(h, w, number_of_workers,
		convolution<float, components<3, 32>, kernel<3>, stride<2>, padding<1>>(weight1),
		batch_norm_relu<float, components<32>>(bnbeta2, bnmean2, bngamma_variance_epsilon2, 0, 6),
		depthwise_convolution<float, components<32>, kernel<3>, stride<1>, padding<2>>(weight3),
		batch_norm_relu<float, components<32>>(bnbeta4, bnmean4, bngamma_variance_epsilon4, 0, 6),
		pointwise_convolution<float, components<32, 64>>(weight5),
		batch_norm_relu<float, components<64>>(bnbeta6, bnmean6, bngamma_variance_epsilon6, 0, 6),
		depthwise_convolution<float, components<64>, kernel<3>, stride<2>, padding<1>>(weight7),
		batch_norm_relu<float, components<64>>(bnbeta8, bnmean8, bngamma_variance_epsilon8, 0, 6),
		pointwise_convolution<float, components<64, 128>>(weight9),
		batch_norm_relu<float, components<128>>(bnbeta10, bnmean10, bngamma_variance_epsilon10, 0, 6),
		depthwise_convolution<float, components<128>, kernel<3>, stride<1>, padding<2>>(weight11),
		batch_norm_relu<float, components<128>>(bnbeta12, bnmean12, bngamma_variance_epsilon12, 0, 6),
		pointwise_convolution<float, components<128>>(weight13),
		batch_norm_relu<float, components<128>>(bnbeta14, bnmean14, bngamma_variance_epsilon14, 0, 6),
		depthwise_convolution<float, components<128>, kernel<3>, stride<2>, padding<1>>(weight15),
		batch_norm_relu<float, components<128>>(bnbeta16, bnmean16, bngamma_variance_epsilon16, 0, 6),
		pointwise_convolution<float, components<128, 256>>(weight17),
		batch_norm_relu<float, components<256>>(bnbeta18, bnmean18, bngamma_variance_epsilon18, 0, 6),
		depthwise_convolution<float, components<256>, kernel<3>, stride<1>, padding<2>>(weight19),
		batch_norm_relu<float, components<256>>(bnbeta20, bnmean20, bngamma_variance_epsilon20, 0, 6),
		pointwise_convolution<float, components<256>>(weight21),
		batch_norm_relu<float, components<256>>(bnbeta22, bnmean22, bngamma_variance_epsilon22, 0, 6),
		depthwise_convolution<float, components<256>, kernel<3>, stride<2>, padding<1>>(weight23),
		batch_norm_relu<float, components<256>>(bnbeta24, bnmean24, bngamma_variance_epsilon24, 0, 6),
		pointwise_convolution<float, components<256, 512>>(weight25),
		batch_norm_relu<float, components<512>>(bnbeta26, bnmean26, bngamma_variance_epsilon26, 0, 6),
		depthwise_convolution<float, components<512>, kernel<3>, stride<1>, padding<2>>(weight27),
		batch_norm_relu<float, components<512>>(bnbeta28, bnmean28, bngamma_variance_epsilon28, 0, 6),
		pointwise_convolution<float, components<512>>(weight29),
		batch_norm_relu<float, components<512>>(bnbeta30, bnmean30, bngamma_variance_epsilon30, 0, 6),
		depthwise_convolution<float, components<512>, kernel<3>, stride<1>, padding<2>>(weight31),
		batch_norm_relu<float, components<512>>(bnbeta32, bnmean32, bngamma_variance_epsilon32, 0, 6),
		pointwise_convolution<float, components<512>>(weight33),
		batch_norm_relu<float, components<512>>(bnbeta34, bnmean34, bngamma_variance_epsilon34, 0, 6),
		depthwise_convolution<float, components<512>, kernel<3>, stride<1>, padding<2>>(weight35),
		batch_norm_relu<float, components<512>>(bnbeta36, bnmean36, bngamma_variance_epsilon36, 0, 6),
		pointwise_convolution<float, components<512>>(weight37),
		batch_norm_relu<float, components<512>>(bnbeta38, bnmean38, bngamma_variance_epsilon38, 0, 6),
		depthwise_convolution<float, components<512>, kernel<3>, stride<1>, padding<2>>(weight39),
		batch_norm_relu<float, components<512>>(bnbeta40, bnmean40, bngamma_variance_epsilon40, 0, 6),
		pointwise_convolution<float, components<512>>(weight41),
		batch_norm_relu<float, components<512>>(bnbeta42, bnmean42, bngamma_variance_epsilon42, 0, 6),
		depthwise_convolution<float, components<512>, kernel<3>, stride<1>, padding<2>>(weight43),
		batch_norm_relu<float, components<512>>(bnbeta44, bnmean44, bngamma_variance_epsilon44, 0, 6),
		pointwise_convolution<float, components<512>>(weight45),
		batch_norm_relu<float, components<512>>(bnbeta46, bnmean46, bngamma_variance_epsilon46, 0, 6),
		depthwise_convolution<float, components<512>, kernel<3>, stride<2>, padding<1>>(weight47),
		batch_norm_relu<float, components<512>>(bnbeta48, bnmean48, bngamma_variance_epsilon48, 0, 6),
		pointwise_convolution<float, components<512, 1024>>(weight49),
		batch_norm_relu<float, components<1024>>(bnbeta50, bnmean50, bngamma_variance_epsilon50, 0, 6),
		depthwise_convolution<float, components<1024>, kernel<3>, stride<1>, padding<2>>(weight51),
		batch_norm_relu<float, components<1024>>(bnbeta52, bnmean52, bngamma_variance_epsilon52, 0, 6),
		pointwise_convolution<float, components<1024>>(weight53),
		batch_norm_relu<float, components<1024>>(bnbeta54, bnmean54, bngamma_variance_epsilon54, 0, 6),
		avg_pooling<float, components<1024>, kernel<7>, stride<2>>(),
		pointwise_convolution<float, components<1024, 1001>>(weight55),
		bias<float, components<1001>>(weight56),
		softmax<float, components<1001>>()
	);
}
