program xgboost_serialization_probe
    !! Release probe for XGBoost text save/load and prediction equivalence.
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    implicit none

    type(xgboost_t) :: original, restored
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status
    real(real64) :: x(10, 2), y(10), validation_y(10), query(7, 2)
    real(real64) :: before(7), after(7), margins_before(7), margins_after(7)
    real(real64), allocatable :: staged_before(:, :), staged_after(:, :)
    character(len=512) :: path
    integer :: i, j, estimator_count

    path = "xgboost_release_model.txt"
    call get_command_argument(1, path)
    if (len_trim(path) == 0) path = "xgboost_release_model.txt"
    do i = 1, 10
        x(i, 1) = real(i - 1, real64)
        x(i, 2) = real(mod(3*i + 1, 7), real64) - 2.0_real64
        y(i) = merge(4.0_real64, -1.0_real64, i > 5)
        validation_y(i) = 0.5_real64*y(i)
    end do
    x(3, 2) = ieee_value(x(3, 2), ieee_quiet_nan)
    query(:, 1) = [-1.0_real64, 0.0_real64, 1.5_real64, 3.5_real64, 5.0_real64, 7.0_real64, 9.0_real64]
    query(:, 2) = [-2.0_real64, -1.0_real64, 0.0_real64, 1.0_real64, 2.0_real64, -2.0_real64, 1.0_real64]

    options = xgboost_options_t()
    options%n_estimators = 4
    options%max_depth = 2
    options%learning_rate = 0.6_real64
    options%missing_policy = "learn"
    options%subsample = 0.8_real64
    options%colsample_bytree = 0.5_real64
    options%seed = 71
    options%early_stopping_rounds = 2
    options%restore_best = .false.
    options%monotone_constraints = [1, 0]
    call original%fit_regression(x, y, status, options, validation_x=x, validation_y=validation_y)
    if (.not. status_ok(status)) error stop "XGBoost fit failed"
    estimator_count = original%estimator_count()
    allocate(staged_before(7, estimator_count), staged_after(7, estimator_count))
    call original%predict(query, before, status)
    if (.not. status_ok(status)) error stop "XGBoost prediction failed"
    call original%predict_margin(query, margins_before, status)
    if (.not. status_ok(status)) error stop "XGBoost margin failed"
    call original%predict_staged(query, staged_before, status)
    if (.not. status_ok(status)) error stop "XGBoost staged prediction failed"
    call original%save_text(trim(path), status)
    if (.not. status_ok(status)) error stop "XGBoost save failed"
    call restored%load_text(trim(path), status)
    if (.not. status_ok(status)) error stop "XGBoost load failed"
    call restored%predict(query, after, status)
    if (.not. status_ok(status)) error stop "XGBoost restored prediction failed"
    call restored%predict_margin(query, margins_after, status)
    if (.not. status_ok(status)) error stop "XGBoost restored margin failed"
    call restored%predict_staged(query, staged_after, status)
    if (.not. status_ok(status)) error stop "XGBoost restored staged prediction failed"

    do i = 1, 7
        write (*, '(a,i0,a,es26.17e3)') "xgb_serialization_prediction_before,", i, ",", before(i)
        write (*, '(a,i0,a,es26.17e3)') "xgb_serialization_prediction_after,", i, ",", after(i)
        write (*, '(a,i0,a,es26.17e3)') "xgb_serialization_margin_before,", i, ",", margins_before(i)
        write (*, '(a,i0,a,es26.17e3)') "xgb_serialization_margin_after,", i, ",", margins_after(i)
        do j = 1, estimator_count
            write (*, '(a,2(i0,a),es26.17e3)') "xgb_serialization_staged_before,", i, ",", j, ",", staged_before(i, j)
            write (*, '(a,2(i0,a),es26.17e3)') "xgb_serialization_staged_after,", i, ",", j, ",", staged_after(i, j)
        end do
    end do
    write (*, '(a,i0)') "xgb_serialization_estimator_count,", estimator_count
    write (*, '(a,i0)') "xgb_serialization_best_iteration,", original%best_iteration()
    write (*, '(a,es26.17e3)') "xgb_serialization_best_loss,", original%best_validation_loss()
    write (*, '(a,i0)') "xgb_serialization_monotone_1,", original%monotone_constraint(1)
    write (*, '(a,i0)') "xgb_serialization_monotone_2,", original%monotone_constraint(2)
    write (*, '(a,i0)') "xgb_serialization_cuda,3"
end program xgboost_serialization_probe
