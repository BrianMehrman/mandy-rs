#[macro_use]
extern crate clap;
extern crate find_folder;
extern crate image;
extern crate ocl;

use clap::{Arg, App, AppSettings};
use ocl::{flags, Buffer, Context, Queue, Device, Platform, Program, Kernel, Image};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType};
use find_folder::Search;

//#[allow(dead_code)]
fn run() -> ocl::Result<()> {

    let matches = App::new("mandy")
        .version("v0.1")
        .setting(AppSettings::AllowNegativeNumbers)
        .arg(Arg::with_name("width").short("w").takes_value(true))
        .arg(Arg::with_name("height").short("h").takes_value(true))
        .arg(Arg::with_name("mid_x").short("x").takes_value(true))
        .arg(Arg::with_name("mid_y").short("y").takes_value(true))
        .arg(Arg::with_name("zoom").short("z").takes_value(true))
        .arg(Arg::with_name("max").short("m").takes_value(true))
        .get_matches();

    let width = value_t!(matches, "width", u32).unwrap_or(1024);
    let height = value_t!(matches, "height", u32).unwrap_or(600);
    let mid_x = value_t!(matches, "mid_x", f64).unwrap_or(0.75);
    let mid_y = value_t!(matches, "mid_y", f64).unwrap_or(0.0);
    let zoom = value_t!(matches, "zoom", f64).unwrap_or(1.0);
    let max = value_t!(matches, "max", u32).unwrap_or(25);

    let dims = (width * height) as usize;
    let mut x_vec = vec![0.0f64; dims];
    let mut y_vec = vec![0.0f64; dims];

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

    let x_buffer = unsafe { 
        Buffer::<f64>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(dims)
        .use_host_slice(&x_vec)
        .build().unwrap()
    };

    let y_buffer = unsafe { 
        Buffer::<f64>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(dims)
            .use_host_slice(&y_vec)
            .build()
            .unwrap()
    };

    let mut img = image::ImageBuffer::from_pixel(width, height, image::Rgba([0, 0, 0, 255u8]));
    let dst_image = unsafe {
        Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&img.dimensions())
            .use_host_slice(&img)
            .queue(queue.clone())
            .build().unwrap()
    };

    let kernel = Kernel::builder()
        .program(&program)
        .queue(queue.clone())
        .global_work_size(dims)
        .name("mandy").arg(&x_buffer).arg(&y_buffer).arg(&max).arg(&width).arg(&dst_image)
        .build()?;

    // fill x/y to Mandelbrot coordinates
    let height_vs_width = height as f64 / width as f64;
    let scale_h = height as f64 * 0.5 / height as f64 * 3.5 * height_vs_width * zoom;
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

    unsafe { kernel.enq()? }
    dst_image.read(&mut img).enq().unwrap();

    let mut path = std::env::current_dir().unwrap();
    path.push("result.png");
    println!("saving image to {}", path.display());
    img.save(path.to_str().unwrap()).unwrap();
    Ok(())
}

fn main() {
    run().unwrap();
}
