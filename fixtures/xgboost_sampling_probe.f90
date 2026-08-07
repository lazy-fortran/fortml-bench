program xgboost_sampling_probe
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 12, p = 4
    real(dp) :: x(n, p), y(n), prediction(n), importance(p)
    type(xgboost_t) :: model
    type(xgboost_options_t) :: options
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    integer :: i

    do i = 1, n
        x(i, 1) = real(i - 1, dp)
        x(i, 2) = real(mod(3*i + 1, 11), dp)
        x(i, 3) = real(mod(5*i + 2, 13), dp)
        x(i, 4) = real(mod(7*i + 3, 17), dp)
        y(i) = 1.4_dp*x(i, 1) - 0.55_dp*x(i, 2) + 0.2_dp*x(i, 3) + &
            0.1_dp*x(i, 4) + merge(0.7_dp, -0.4_dp, mod(i, 2) == 0)
    end do
    options%n_estimators = 1
    options%max_depth = 1
    options%learning_rate = 0.8_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp
    options%subsample = 0.5_dp
    options%colsample_bytree = 0.5_dp
    options%seed = 12345_int64
    call model%fit_regression(x, y, status, options)
    if (.not. status_ok(status)) error stop "xgboost sampling probe fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "xgboost sampling probe prediction failed"
    call model%feature_importance(importance, status, kind="weight")
    if (.not. status_ok(status)) error stop "xgboost sampling probe importance failed"
    write (*, '(a,es24.16)') "xgb_sampling_base,", model%base_margin()
    write (*, '(a,i0)') "xgb_sampling_node_count,", model%tree_node_count(1)
    write (*, '(a,i0)') "xgb_sampling_depth,", model%tree_depth(1)
    write (*, '(a,4(es24.16,:,a))') "xgb_sampling_importance,", importance(1), ",", &
        importance(2), ",", importance(3), ",", importance(4)
    write (*, '(a,12(es24.16,:,a))') "xgb_sampling_prediction,", prediction(1), ",", &
        prediction(2), ",", prediction(3), ",", prediction(4), ",", prediction(5), ",", &
        prediction(6), ",", prediction(7), ",", prediction(8), ",", prediction(9), ",", &
        prediction(10), ",", prediction(11), ",", prediction(12)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    write (*, '(a,i0)') "xgb_sampling_cuda,", status%code
end program xgboost_sampling_probe
