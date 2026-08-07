program mlp_multilabel_classifier_probe
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_mlp_multilabel_classifier, only: mlp_multilabel_classifier_t, &
        mlp_multilabel_classifier_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 6, p = 2, labels_count = 2
    real(dp) :: x(n, p), targets(n, labels_count)
    real(dp) :: probabilities(n, labels_count), loss
    real(dp), allocatable :: theta(:), gradient(:), direction(:), hvp(:)
    integer :: predicted(n, labels_count), i, j
    type(mlp_multilabel_classifier_t) :: model
    type(mlp_multilabel_classifier_options_t) :: options
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x(:, 1) = [-1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.2_dp]
    x(:, 2) = [-1.0_dp, -0.2_dp, 0.0_dp, 0.2_dp, 1.0_dp, 0.8_dp]
    targets(:, 1) = [0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
    targets(:, 2) = [0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp]
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

    call model%fit(x, targets, status, options=options)
    if (.not. status_ok(status)) error stop "multilabel MLP probe fit failed"
    theta = model%parameters()
    allocate(direction(size(theta)), hvp(size(theta)), gradient(size(theta)))
    do i = 1, size(theta)
        direction(i) = 0.01_dp*real(i, dp)
    end do
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    call model%loss_gradient(x, targets, options%l2, loss, gradient, status)
    call model%loss_hvp(x, targets, options%l2, direction, hvp, status)
    if (.not. status_ok(status)) error stop "multilabel MLP probe derivative failed"
    write (*, '(a,i0)') "mlp_multilabel_parameter_count,", size(theta)
    write (*, '(a,es24.16)') "mlp_multilabel_loss,", loss
    do i = 1, size(theta)
        write (*, '(a,i0,a,es24.16)') "mlp_multilabel_theta,", i, ",", theta(i)
        write (*, '(a,i0,a,es24.16)') "mlp_multilabel_gradient,", i, ",", gradient(i)
        write (*, '(a,i0,a,es24.16)') "mlp_multilabel_hvp,", i, ",", hvp(i)
    end do
    do i = 1, n
        do j = 1, labels_count
            write (*, '(a,2(i0,a),es24.16)') "mlp_multilabel_probability,", i, ",", &
                j, ",", probabilities(i, j)
            write (*, '(a,2(i0,a),i0)') "mlp_multilabel_prediction,", i, ",", &
                j, ",", predicted(i, j)
        end do
    end do
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    write (*, '(a,i0)') "mlp_multilabel_cuda,", status%code
end program mlp_multilabel_classifier_probe
