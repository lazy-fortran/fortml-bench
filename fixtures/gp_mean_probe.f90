program gp_mean_probe
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_gp_mean, only: gp_mean_t, make_constant_mean, make_linear_mean
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 4, q = 2
    real(dp) :: x(n, 1), y(n, 1), query(q, 1)
    real(dp) :: mean(q, 1), variance(q), lml
    real(dp) :: gradient(4), direction(4), hvp(4)
    real(dp) :: mean_linear(q, 1), variance_linear(q), lml_linear
    real(dp) :: gradient_linear(5), direction_linear(5), hvp_linear(5)
    real(dp), allocatable :: parameters(:), parameters_linear(:)
    type(gp_regression_t) :: constant_model, linear_model
    type(gp_mean_t) :: constant_mean, linear_mean
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    integer :: i

    x(:, 1) = [-1.0_dp, -1.0_dp/3.0_dp, 1.0_dp/3.0_dp, 1.0_dp]
    do i = 1, n
        y(i, 1) = 0.3_dp + 0.2_dp*x(i, 1) + 0.05_dp*x(i, 1)**2
    end do
    query(:, 1) = [-0.5_dp, 1.5_dp]
    kernel = make_rbf_kernel(1, 1.3_dp, 0.8_dp, status)
    constant_mean = make_constant_mean(1, status, 0.2_dp)
    call constant_model%fit(x, y, kernel, 0.15_dp, status, mean=constant_mean)
    if (.not. status_ok(status)) error stop "constant GP mean probe fit failed"
    parameters = constant_model%parameters()
    direction = [0.0_dp, 0.0_dp, 0.0_dp, 0.07_dp]
    call constant_model%predict(query, mean, variance, status)
    call constant_model%log_marginal_likelihood(lml, status)
    call constant_model%hyperparameter_gradient(gradient, status)
    call constant_model%hyperparameter_hvp(direction, hvp, status)
    if (.not. status_ok(status)) error stop "constant GP mean probe derivatives failed"
    write (*, '(a,i0)') "gp_mean_constant_parameter_count,", size(parameters)
    write (*, '(a,i0)') "gp_mean_constant_mean_parameter_count,", constant_model%mean_parameter_count()
    write (*, '(a,es24.16)') "gp_mean_constant_lml,", lml
    do i = 1, size(parameters)
        write (*, '(a,i0,a,es24.16)') "gp_mean_constant_parameter,", i, ",", parameters(i)
        write (*, '(a,i0,a,es24.16)') "gp_mean_constant_gradient,", i, ",", gradient(i)
        write (*, '(a,i0,a,es24.16)') "gp_mean_constant_hvp,", i, ",", hvp(i)
    end do
    do i = 1, q
        write (*, '(a,i0,a,es24.16,a,es24.16)') "gp_mean_constant_prediction,", i, ",", &
            mean(i, 1), ",", variance(i)
    end do

    linear_mean = make_linear_mean(1, status, [0.1_dp, 0.5_dp])
    call linear_model%fit(x, y, kernel, 0.15_dp, status, mean=linear_mean)
    if (.not. status_ok(status)) error stop "linear GP mean probe fit failed"
    parameters_linear = linear_model%parameters()
    direction_linear = [0.0_dp, 0.0_dp, 0.0_dp, 0.02_dp, -0.03_dp]
    call linear_model%predict(query, mean_linear, variance_linear, status)
    call linear_model%log_marginal_likelihood(lml_linear, status)
    call linear_model%hyperparameter_gradient(gradient_linear, status)
    call linear_model%hyperparameter_hvp(direction_linear, hvp_linear, status)
    if (.not. status_ok(status)) error stop "linear GP mean probe derivatives failed"
    write (*, '(a,i0)') "gp_mean_linear_parameter_count,", size(parameters_linear)
    write (*, '(a,i0)') "gp_mean_linear_mean_parameter_count,", linear_model%mean_parameter_count()
    write (*, '(a,es24.16)') "gp_mean_linear_lml,", lml_linear
    do i = 1, size(parameters_linear)
        write (*, '(a,i0,a,es24.16)') "gp_mean_linear_parameter,", i, ",", parameters_linear(i)
        write (*, '(a,i0,a,es24.16)') "gp_mean_linear_gradient,", i, ",", gradient_linear(i)
        write (*, '(a,i0,a,es24.16)') "gp_mean_linear_hvp,", i, ",", hvp_linear(i)
    end do
    do i = 1, q
        write (*, '(a,i0,a,es24.16,a,es24.16)') "gp_mean_linear_prediction,", i, ",", &
            mean_linear(i, 1), ",", variance_linear(i)
    end do
    write (*, '(a,i0)') "gp_mean_cuda,", 3
end program gp_mean_probe
