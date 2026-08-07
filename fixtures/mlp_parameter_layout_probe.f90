program mlp_parameter_layout_probe
    use fortml_mlp, only: mlp_t, mlp_parameter_block_t, MLP_TANH, MLP_LINEAR
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(mlp_t) :: model
    type(mlp_parameter_block_t), allocatable :: layout(:)
    type(fortnum_status_t) :: status
    integer :: i, first, last
    logical :: found

    call model%initialize([3, 4, 2], status, hidden_activation=MLP_TANH, &
        output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop "MLP layout probe: initialize failed"
    layout = model%parameter_layout()
    write (*, '("parameter_count,",i0)') model%parameter_count()
    write (*, '("parameter_block_count,",i0)') model%parameter_block_count()
    do i = 1, size(layout)
        write (*, '("layout,",i0,",",a,",",a,",",i0,",",i0,",",i0,",",i0,",",l1,",",l1)') &
            i, trim(layout(i)%name), trim(layout(i)%kind), layout(i)%first, layout(i)%last, &
            layout(i)%rows, layout(i)%columns, layout(i)%trainable, layout(i)%is_buffer
    end do
    call model%parameter_range("layer_2.weight", first, last, found)
    write (*, '("range,layer_2.weight,",i0,",",i0,",",l1)') first, last, found
end program mlp_parameter_layout_probe
