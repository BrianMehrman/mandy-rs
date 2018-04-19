extern crate clap;
extern crate find_folder;
extern crate image;
extern crate ocl;

use ocl::{flags, Buffer, Context, Queue, Device, Platform, Program, Kernel, Image};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType};
use find_folder::Search;

//#[allow(dead_code)]
fn trivial_exploded() -> ocl::Result<()> {

    let kernel_src = Search::ParentsThenKids(3, 3)
        .for_folder("kernel")
        .expect("Error locating 'kernel'")
        .join("mandy.cl");

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims (compare to step 1 above).
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;
    let program = Program::builder()
        .devices(device)
        .src_file(kernel_src)
        .build(&context)?;
    let queue = Queue::new(&context, device, None)?;

    let width = 1920u32;
    let height = 1080u32;
    let mid_x = 0.75f64;
    let mid_y = 0.0f64;
    let zoom = 1.0f64;
    let max = 50;

    let mut img = image::ImageBuffer::from_pixel(width, height, image::Rgba([0, 0, 0, 255u8]));// generate_image(width, height);
    let dims = (width * height) as usize;

    let dst_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&img.dimensions())
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(&img)
        .queue(queue.clone())
        .build().unwrap();

    let mut x_vec = vec![0.0f64; dims];
    let mut y_vec = vec![0.0f64; dims];

    // fill x/y to Mandelbrot coordinates
    let scale_h = height as f64 * 0.5 / height as f64 * 2.0 * zoom;
    let scale_w =  width as f64 * 0.5 /  width as f64 * 3.5 * zoom;

    let left = -scale_w - mid_x;
    let right = scale_w - mid_x;
    let top = -scale_h - mid_y;
    let bottom = scale_h - mid_y;
    let step_x = (right - left) / (width as f64 - 1.0);
    let step_y = (bottom - top) / (height as f64 - 1.0);

    let mut y = top;
    for h in 0..height {
        let mut x = left;
        for w in 0..width {
            let offset = (h * width + w) as usize;
            x_vec[offset] = x;
            y_vec[offset] = y;
            x += step_x;
        }
        y += step_y;
    }

    let x_buffer = Buffer::<f64>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(dims)
        .copy_host_slice(&x_vec)
        .build()?;

    let y_buffer = Buffer::<f64>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(dims)
        .copy_host_slice(&y_vec)
        .build()?;

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = Kernel::builder()
        .program(&program)
        .name("mandy")
        .queue(queue.clone())
        .global_work_size(dims)
        .arg(&x_buffer)
        .arg(&y_buffer)
        .arg(&max)
        .arg(&width)
        .arg(&dst_image)
        .build()?;

    // (4) Run the kernel (default parameters shown for demonstration purposes):
    println!("running kernel");
    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(kernel.default_local_work_size())
            .enq()?;
    }

    println!("read into image");
    dst_image.read(&mut img).enq().unwrap();

    let mut path = std::env::current_dir().unwrap();
    path.push("result.png");
    println!("saving image to {}", path.display());
    img.save(path.to_str().unwrap()).unwrap();
    Ok(())
}

fn main() {
    trivial_exploded().unwrap();
}
