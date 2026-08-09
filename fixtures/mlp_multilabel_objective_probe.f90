program mlp_multilabel_objective_probe
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_mlp_multilabel_classifier, only: &
        mlp_multilabel_classifier_t, mlp_multilabel_classifier_options_t, &
        mlp_multilabel_training_objective_t, mlp_multilabel_lbfgsb_options_t, &
        mlp_multilabel_lbfgsb_result_t, mlp_multilabel_optimize_lbfgsb
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 6, p = 2, label_count = 2
    real(dp) :: x(n, p), targets(n, label_count), sample_weight(n)
    real(dp) :: probabilities(n, label_count)
    real(dp) :: class_weight(2, label_count), direction(7)
    real(dp), allocatable :: theta(:), parameters(:), gradient(:), product(:), vjp_product(:)
    real(dp), allocatable :: log_parameters(:), log_gradient(:), log_product(:)
    real(dp) :: value, tangent, vjp_scale, optimizer_value
    integer :: indicators(n, label_count), i
    type(mlp_multilabel_classifier_t), target :: model, log_model
    type(mlp_multilabel_classifier_options_t) :: fit_options
    type(mlp_multilabel_training_objective_t) :: objective, log_objective
    type(mlp_multilabel_lbfgsb_options_t) :: options
    type(mlp_multilabel_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda

    x(:, 1) = [-1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.2_dp]
    x(:, 2) = [-1.0_dp, -0.2_dp, 0.0_dp, 0.2_dp, 1.0_dp, 0.8_dp]
    indicators(:, 1) = [0, 0, 0, 1, 1, 1]
    indicators(:, 2) = [0, 1, 0, 1, 0, 1]
    targets = real(indicators, dp)
    sample_weight = [0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp, 0.75_dp, 1.25_dp]
    class_weight = reshape([1.1_dp, 0.8_dp, 0.9_dp, 1.4_dp], shape(class_weight))
    fit_options%max_epochs = 1
    fit_options%learning_rate = 0.03_dp
    fit_options%beta1 = 0.8_dp
    fit_options%beta2 = 0.95_dp
    fit_options%epsilon = 1.0e-7_dp
    fit_options%l2 = 0.02_dp
    fit_options%tolerance = 0.0_dp
    fit_options%restore_best = .false.
    fit_options%initialization_seed = 29
    call model%fit(x, indicators, status, options=fit_options)
    if (.not. status_ok(status)) error stop "multilabel objective fixture fit failed"
    theta = model%parameters()
    direction = 0.01_dp*[(real(i, dp), i=1, 7)]
    call objective%initialize(model, x, indicators, 0.02_dp, status, &
        optimize_l2=.true., sample_weight=sample_weight, class_weight=class_weight)
    if (.not. status_ok(status)) error stop "multilabel objective initialization failed"
    parameters = objective%parameters()
    allocate(gradient(size(parameters)))
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "multilabel objective gradient failed"
    call objective%jvp(parameters, direction, optimizer_value, tangent, status)
    if (.not. status_ok(status)) error stop "multilabel objective JVP failed"
    vjp_scale = 1.7_dp
    allocate(product(size(parameters)), vjp_product(size(parameters)))
    call objective%vjp(parameters, vjp_scale, vjp_product, status)
    if (.not. status_ok(status)) error stop "multilabel objective VJP failed"
    call objective%hvp(parameters, direction, product, status)
    if (.not. status_ok(status)) error stop "multilabel objective HVP failed"
    write (*, '(a,i0)') "objective_parameter_count,", size(parameters)
    call emit_vector("objective_theta", parameters)
    call emit_vector("objective_gradient", gradient)
    call emit_vector("objective_hvp", product)
    write (*, '(a,es24.16)') "objective_value,", value
    write (*, '(a,es24.16)') "objective_jvp,", tangent
    write (*, '(a,es24.16)') "objective_vjp_dot,", dot_product(vjp_product, direction)

    call log_model%fit(x, indicators, status, options=fit_options)
    if (.not. status_ok(status)) error stop "multilabel log fixture fit failed"
    call log_objective%initialize(log_model, x, indicators, 0.02_dp, status, &
        optimize_log_l2=.true., sample_weight=sample_weight, class_weight=class_weight)
    if (.not. status_ok(status)) error stop "multilabel log objective initialization failed"
    log_parameters = log_objective%parameters()
    allocate(log_gradient(size(log_parameters)), log_product(size(log_parameters)))
    call log_objective%value_gradient(log_parameters, value, log_gradient, status)
    if (.not. status_ok(status)) error stop "multilabel log objective gradient failed"
    call log_objective%hvp(log_parameters, direction, log_product, status)
    if (.not. status_ok(status)) error stop "multilabel log objective HVP failed"
    write (*, '(a,i0)') "log_objective_parameter_count,", size(log_parameters)
    call emit_vector("log_objective_parameters", log_parameters)
    call emit_vector("log_objective_gradient", log_gradient)
    call emit_vector("log_objective_hvp", log_product)
    write (*, '(a,es24.16)') "log_objective_value,", value

    options%max_iterations = 500
    options%max_line_search = 80
    options%gradient_tolerance = 1.0e-6_dp
    options%step_tolerance = 1.0e-9_dp
    options%objective_tolerance = 1.0e-8_dp
    options%l2 = 0.02_dp
    options%optimize_l2 = .true.
    options%l2_lower_bound = 0.02_dp
    options%l2_upper_bound = 1.0_dp
    call mlp_multilabel_optimize_lbfgsb(model, x, indicators, options, result, status, &
        sample_weight=sample_weight, class_weight=class_weight)
    write (*, '(a,i0,a,l1,a,es24.16,a,es24.16,a,es24.16)') &
        "direct_optimizer,", status%code, ",", result%converged, ",", &
        result%objective, ",", result%gradient_norm, ",", result%l2
    options%optimize_l2 = .false.
    options%optimize_log_l2 = .true.
    options%gradient_tolerance = 1.0e-2_dp
    options%log_l2_lower_bound = -8.0_dp
    options%log_l2_upper_bound = 1.0_dp
    call mlp_multilabel_optimize_lbfgsb(log_model, x, indicators, options, result, status, &
        sample_weight=sample_weight, class_weight=class_weight)
    write (*, '(a,i0,a,l1,a,es24.16,a,es24.16,a,es24.16)') &
        "log_optimizer,", status%code, ",", result%converged, ",", &
        result%objective, ",", result%gradient_norm, ",", result%l2
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    write (*, '(a,i0)') "cuda_status,", status%code

contains

    subroutine emit_vector(name, values)
        character(*), intent(in) :: name
        real(dp), intent(in) :: values(:)
        integer :: k

        do k = 1, size(values)
            write (*, '(a,i0,a,es24.16)') trim(name) // ",", k, ",", values(k)
        end do
    end subroutine emit_vector

end program mlp_multilabel_objective_probe
