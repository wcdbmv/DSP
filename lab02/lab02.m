function lab02
	t_max = 5;
	t = -t_max:0.05:t_max;

	rect = rectangular(t);
	gauss = gaussian(t);

	rect_fft = fft(rect);
	gauss_fft = fft(gauss);

	rect_fft_notwin = fftshift(rect_fft);
	gauss_fft_notwin = fftshift(gauss_fft);

	rect_dft = dft(rect);
	gauss_dft = dft(gauss);

	rect_dft_notwin = fftshift(rect_dft);
	gauss_dft_notwin = fftshift(gauss_dft);

	xs = 0:length(t)-1;

	graph_figure(1, xs, rect_fft, rect_fft_notwin, 'БПФ', 'прямоугольного сигнала')
	graph_figure(2, xs, gauss_fft, gauss_fft_notwin, 'БПФ', 'сигнала Гаусса')
	graph_figure(3, xs, rect_dft, rect_dft_notwin, 'ДПФ', 'прямоугольного сигнала')
	graph_figure(4, xs, gauss_dft, gauss_dft_notwin, 'ДПФ', 'сигнала Гаусса')
end

function y = gaussian(x)
	A = 1;
	sigma = 0.5;

	y = A * exp(-(x / sigma) .^ 2);
end

function y = rectangular(x)
	c = 2;

	y = zeros(size(x));
	y(abs(x) < c) = 1;
	y(abs(x) == c) = 0.5;
end

function y = dft(x)
	a = 0:length(x)-1;
	b = -2 * pi * sqrt(-1) * a / length(x);
	for i = 1:length(a)
		a(i) = 0;
		for j = 1:length(x)
			a(i) = a(i) + x(j) * exp(b(i) * j);
		end
	end
	y = a;
end

function graph_figure(i, xs, y, y_notwin, alg, pulse_name)
	figure(i);
	graph_subplot(1, xs, y, y_notwin, @(x) abs(x), alg, 'амплитудный', pulse_name);
	graph_subplot(2, xs, y, y_notwin, @(x) angle(x), alg, 'фазовый', pulse_name);
	print(strcat('plot02_', num2str(i)), '-dpng');
end

function graph_subplot(i, xs, y, y_notwin, f, alg, spectrum_name, pulse_name)
	subplot(2, 1, i);
	plot(xs, f(y) / length(xs), xs, f(y_notwin) / length(xs));
	title([alg, ': ', spectrum_name, ' спектр ', pulse_name]);
	legend('С эффектом «близнецов»', 'без эффекта «близнецов»');
end
