program mlp_binary_classifier_probe
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_mlp_binary_classifier, only: mlp_binary_classifier_t, &
        mlp_binary_classifier_options_t, mlp_binary_classifier_state_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 6, p = 2
    real(dp) :: x(n, p), scores(n), probabilities(n, 2)
    real(dp), allocatable :: theta(:), gradient(:), direction(:), hvp(:)
    real(dp) :: loss
    integer :: labels(n), predicted(n), classes(2), i
    type(mlp_binary_classifier_t) :: model
    type(mlp_binary_classifier_options_t) :: options
    type(mlp_binary_classifier_state_t) :: state
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x(:, 1) = [-1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.2_dp]
    x(:, 2) = [-1.0_dp, -0.2_dp, 0.0_dp, 0.2_dp, 1.0_dp, 0.8_dp]
    labels = [-2, -2, -2, 5, 5, 5]
    options%max_epochs = 1
    options%batch_size = 0
    options%patience = 0
    options%restore_best = .false.
    options%learning_rate = 0.03_dp
    options%beta1 = 0.8_dp
    options%beta2 = 0.95_dp
    options%epsilon = 1.0e-7_dp
    options%l2 = 0.02_dp
    options%tolerance = 0.0_dp
    options%initialization_seed = 29
    call model%fit(x, labels, status, options=options, state=state)
    if (.not. status_ok(status)) error stop "binary MLP probe fit failed"
    theta = model%parameters()
    allocate(direction(size(theta)), hvp(size(theta)), gradient(size(theta)))
    do i = 1, size(theta)
        direction(i) = 0.01_dp*real(i, dp)
    end do
    call model%decision_function(x, scores, status)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    call model%loss_gradient(x, labels, options%l2, loss, gradient, status)
    call model%loss_hvp(x, labels, options%l2, direction, hvp, status)
    if (.not. status_ok(status)) error stop "binary MLP probe derivative failed"
    classes = model%classes()
    write (*, '(a,i0)') "mlp_binary_parameter_count,", size(theta)
    write (*, '(a,i0)') "mlp_binary_classes_1,", classes(1)
    write (*, '(a,i0)') "mlp_binary_classes_2,", classes(2)
    write (*, '(a,i0)') "mlp_binary_epochs,", state%epochs
    write (*, '(a,i0)') "mlp_binary_updates,", state%updates
    write (*, '(a,es24.16)') "mlp_binary_initial_loss,", state%initial_loss
    write (*, '(a,es24.16)') "mlp_binary_final_loss,", state%final_loss
    write (*, '(a,es24.16)') "mlp_binary_loss,", loss
    do i = 1, size(theta)
        write (*, '(a,i0,a,es24.16)') "mlp_binary_theta,", i, ",", theta(i)
        write (*, '(a,i0,a,es24.16)') "mlp_binary_gradient,", i, ",", gradient(i)
        write (*, '(a,i0,a,es24.16)') "mlp_binary_hvp,", i, ",", hvp(i)
    end do
    do i = 1, n
        write (*, '(a,i0,a,es24.16)') "mlp_binary_score,", i, ",", scores(i)
        write (*, '(a,i0,a,2(es24.16,:,a))') "mlp_binary_probability,", i, ",", &
            probabilities(i, 1), ",", probabilities(i, 2)
        write (*, '(a,i0,a,i0)') "mlp_binary_prediction,", i, ",", predicted(i)
    end do
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    write (*, '(a,i0)') "mlp_binary_cuda,", status%code
end program mlp_binary_classifier_probe
