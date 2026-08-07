program gp_ard_probe
    !! Release probe for ARD RBF values, derivative products, and exact GP use.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_kernels, only: kernel_t, make_rbf_ard_kernel
    use fortml_gaussian_process, only: gp_regression_t
    implicit none

    type(kernel_t) :: kernel
    type(gp_regression_t) :: model
    type(fortnum_status_t) :: status
    real(real64), parameter :: variance = 2.1_real64
    real(real64), parameter :: lengthscales(3) = [0.8_real64, 1.2_real64, 1.6_real64]
    real(real64) :: x1(3, 3), x2(2, 3), matrix(3, 2), matrix_dot(3, 2)
    real(real64) :: matrix_bar(3, 2), parameter_bar(4), parameter_bar_dot(4)
    real(real64) :: direction(4), value, gradient_x1(3), gradient_x2(3)
    real(real64) :: mixed_hessian(3, 3), x_train(1, 3), y_train(1, 1)
    real(real64) :: x_query(1, 3), mean(1, 1), posterior_variance(1)
    real(real64) :: lml, gp_gradient(5)
    integer :: i, j

    x1 = reshape([0.0_real64, 0.5_real64, -0.4_real64, 1.0_real64, 1.2_real64, &
        -0.7_real64, -0.2_real64, 0.9_real64, 0.4_real64], shape(x1))
    x2 = reshape([0.2_real64, -0.1_real64, 0.8_real64, 0.4_real64, -0.6_real64, &
        0.3_real64], shape(x2))
    direction = [0.17_real64, -0.23_real64, 0.11_real64, 0.29_real64]
    matrix_bar = reshape([0.4_real64, -0.2_real64, 0.3_real64, 0.5_real64, &
        -0.7_real64, 0.1_real64], shape(matrix_bar))

    kernel = make_rbf_ard_kernel(3, variance, lengthscales, status)
    if (.not. status_ok(status)) error stop "ARD constructor failed"
    call kernel%matrix(x1, x2, matrix, status)
    if (.not. status_ok(status)) error stop "ARD matrix failed"
    do j = 1, size(x2, 1)
        do i = 1, size(x1, 1)
            write (*, '(a,2(i0,a),es26.17e3)') "gp_ard_matrix,", i, ",", j, ",", matrix(i, j)
        end do
    end do

    call kernel%input_derivatives(x1(2, :), x2(1, :), value, gradient_x1, gradient_x2, &
        mixed_hessian, status)
    if (.not. status_ok(status)) error stop "ARD input derivatives failed"
    write (*, '(a,es26.17e3)') "gp_ard_input_value,", value
    do i = 1, 3
        write (*, '(a,i0,a,es26.17e3)') "gp_ard_input_gradient_x1,", i, ",", gradient_x1(i)
        write (*, '(a,i0,a,es26.17e3)') "gp_ard_input_gradient_x2,", i, ",", gradient_x2(i)
        do j = 1, 3
            write (*, '(a,2(i0,a),es26.17e3)') "gp_ard_input_mixed,", i, ",", j, ",", &
                mixed_hessian(i, j)
        end do
    end do

    call kernel%matrix_jvp(x1, x2, direction, matrix, matrix_dot, status)
    if (.not. status_ok(status)) error stop "ARD matrix JVP failed"
    do j = 1, size(x2, 1)
        do i = 1, size(x1, 1)
            write (*, '(a,2(i0,a),es26.17e3)') "gp_ard_matrix_jvp,", i, ",", j, ",", &
                matrix_dot(i, j)
        end do
    end do
    call kernel%parameter_vjp(x1, x2, matrix_bar, parameter_bar, status)
    if (.not. status_ok(status)) error stop "ARD parameter VJP failed"
    do i = 1, 4
        write (*, '(a,i0,a,es26.17e3)') "gp_ard_parameter_vjp,", i, ",", parameter_bar(i)
    end do
    call kernel%parameter_hvp(x1, x2, matrix_bar, direction, parameter_bar, parameter_bar_dot, status)
    if (.not. status_ok(status)) error stop "ARD parameter HVP failed"
    do i = 1, 4
        write (*, '(a,i0,a,es26.17e3)') "gp_ard_parameter_hvp,", i, ",", parameter_bar_dot(i)
    end do

    x_train = 0.0_real64
    y_train = 2.0_real64
    x_query = reshape([1.0_real64, -0.5_real64, 0.25_real64], shape(x_query))
    kernel = make_rbf_ard_kernel(3, 1.5_real64, [0.8_real64, 1.4_real64, 2.1_real64], status)
    call model%fit(x_train, y_train, kernel, 0.2_real64, status, jitter=0.0_real64)
    if (.not. status_ok(status)) error stop "ARD exact GP fit failed"
    call model%predict(x_query, mean, posterior_variance, status)
    if (.not. status_ok(status)) error stop "ARD exact GP prediction failed"
    write (*, '(a,2(es26.17e3,a),es26.17e3)') "gp_ard_prediction,", mean(1, 1), ",", &
        posterior_variance(1), ",", 0.0_real64
    call model%log_marginal_likelihood(lml, status)
    call model%hyperparameter_gradient(gp_gradient, status)
    write (*, '(a,es26.17e3)') "gp_ard_lml,", lml
    do i = 1, size(gp_gradient)
        write (*, '(a,i0,a,es26.17e3)') "gp_ard_gp_gradient,", i, ",", gp_gradient(i)
    end do
    write (*, '(a,i0)') "gp_ard_parameter_count,", kernel%parameter_count()
    write (*, '(a,i0)') "gp_ard_cuda,3"
end program gp_ard_probe
