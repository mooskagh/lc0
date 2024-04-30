class Lc0File {
    constructor(heap8, heap32) {
        if (new.target === Lc0File) {
            throw new TypeError("Cannot construct Lc0File instances directly");
        }
        this.heap8 = heap8;
        this.heap32 = heap32;
    }

    write(mem, ptr, len) {
        throw new TypeError(`Method 'write' not implemented for ${this.constructor.name} file type.`);
    }

    read(mem, ptr, len) {
        throw new TypeError(`Method 'read' not implemented for ${this.constructor.name} file type.`);
    }

    dispatchRead(iov, iovcnt) {
        let ret = 0;
        for (let i = 0; i < iovcnt; i++) {
            var ptr = this.heap32[iov >> 2];
            var len = this.heap32[(iov + 4) >> 2];
            iov += 8;
            var curr = this.read(this.heap8, ptr, len);
            if (curr < 0) return -1;
            ret += curr;
            if (curr < len) break;
        }
        return ret;
    }

    dispatchWrite(iov, iovcnt) {
        let ret = 0;
        for (let i = 0; i < iovcnt; i++) {
            let ptr = this.heap32[iov >> 2];
            let len = this.heap32[(iov + 4) >> 2];
            iov += 8;
            let curr = this.write(this.heap8, ptr, len);
            if (curr < 0) return -1;
            ret += curr;
        }
        return ret;
    }
};

class Lc0CallbackFile extends Lc0File {
    constructor(heap8, heap32, callback) {
        super(heap8, heap32);
        this.callback = callback;
    }

    write(mem, ptr, len) {
        let privateCopy = new Uint8Array(mem.subarray(ptr, ptr + len));
        if (privateCopy.length > 0) {
            this.callback(privateCopy);
        }
        return len;
    }
}

class Lc0ReadQueueFile extends Lc0File {
    constructor(heap8, heap32) {
        super(heap8, heap32);
        this.buffer = new Uint8Array([117, 99, 105, 10]);
        this.encoder = new TextEncoder();
        this.closed = true;
    }

    read(mem, ptr, len) {
        console.log('read', ptr, len);
        if (this.buffer.length == 0) {
            return this.closed ? -1 : 0;
        }
        let sz = Math.min(len, this.buffer.length);
        mem.set(this.buffer.subarray(0, sz), ptr);
        this.buffer = this.buffer.subarray(sz);
        return sz;
    }

    enqueue(data) {
        let olddata = this.buffer;
        let newdata = this.encoder.encode(data);
        this.buffer = new Uint8Array(olddata.length + newdata.length);
        this.buffer.set(olddata, 0);
        this.buffer.set(newdata, olddata.length);
    }
}

class Lc0 {
    constructor(stdout_callback, stderr_callback) {
        this._initialized = this.#initialize(stdout_callback, stderr_callback);
    }

    async run() {
        await this._initialized;
        this.instance.exports['__main_argc_argv'](0, 0);
    }

    send(data) {
        this.files.get(0).enqueue(data + '\n');
    }

    async #initialize(stdout_callback, stderr_callback) {
        const response = await fetch('/lc0.wasm');
        this.module = new WebAssembly.Module(await response.arrayBuffer());
        const INITIAL_MEMORY = this.module['INITIAL_MEMORY'] || 16777216;
        this.memory = new WebAssembly.Memory({
            initial: INITIAL_MEMORY / 65536,
            maximum: INITIAL_MEMORY / 65536,
            shared: true,
        });
        this.heap8 = new Uint8Array(this.memory.buffer);
        this.heap32 = new Uint32Array(this.memory.buffer);
        this.instance = new WebAssembly.Instance(
            this.module, this.#makeImports());
        this.exports = this.instance.exports;
        this.exports['__wasm_call_ctors']();
        this.files = new Map();
        this.files.set(0, new Lc0ReadQueueFile(this.heap8, this.heap32));
        this.files.set(1, new Lc0CallbackFile(this.heap8, this.heap32, stdout_callback));
        this.files.set(2, new Lc0CallbackFile(this.heap8, this.heap32, stderr_callback));
    }

    #fd_write(fd, iov, iovcnt, pnum) {
        if (!this.files.has(fd)) {
            return -1;
        }
        let num = this.files.get(fd).dispatchWrite(iov, iovcnt);
        this.heap32[pnum >> 2] = num;
        return 0;
    }

    #fd_read(fd, iov, iovcnt, pnum) {
        console.log('fd_read', fd, iov, iovcnt, pnum);
        if (!this.files.has(fd)) {
            return -1;
        }
        let num = this.files.get(fd).dispatchRead(iov, iovcnt);
        this.heap32[pnum >> 2] = num;
        return 0;
    }

    #environ_sizes_get(penviron_count, penviron_buf_size) {
        this.heap32[penviron_count >> 2] = 0;
        this.heap32[penviron_buf_size >> 2] = 0;
        return 0;
    }

    #environ_get(env, buf) {
        return 0;
    }

    #syscall_openat(dirfd, path, flags, mode) {
        console.log('syscall_openat', dirfd, path, flags, mode);
        return -1;
    }

    #makeImports() {
        const imports = {
            memory: this.memory,
            "__assert_fail": (...args) => { console.log("__assert_fail(", ...args); },
            "__emscripten_init_main_thread_js": (...args) => { console.log("__emscripten_init_main_thread_js(", ...args); },
            "__emscripten_thread_cleanup": (...args) => { console.log("__emscripten_thread_cleanup(", ...args); },
            "__pthread_create_js": (...args) => { console.log("__pthread_create_js(", ...args); },
            "__syscall_dup": (...args) => { console.log("__syscall_dup(", ...args); },
            "__syscall_fcntl64": (...args) => { console.log("__syscall_fcntl64(", ...args); },
            "__syscall_fstat64": (...args) => { console.log("__syscall_fstat64(", ...args); },
            "__syscall_getdents64": (...args) => { console.log("__syscall_getdents64(", ...args); },
            "__syscall_ioctl": (...args) => { console.log("__syscall_ioctl(", ...args); },
            "__syscall_lstat64": (...args) => { console.log("__syscall_lstat64(", ...args); },
            "__syscall_mkdirat": (...args) => { console.log("__syscall_mkdirat(", ...args); },
            "__syscall_newfstatat": (...args) => { console.log("__syscall_newfstatat(", ...args); },
            "__syscall_openat": (dirfd, path, flags, mode) => { return this.#syscall_openat(dirfd, path, flags, mode); },
            "__syscall_stat64": (...args) => { console.log("__syscall_stat64(", ...args); },
            "__throw_exception_with_stack_trace": (...args) => { console.log("__throw_exception_with_stack_trace(", ...args); },
            "_emscripten_get_now_is_monotonic": () => 1,
            "_emscripten_notify_mailbox_postmessage": (...args) => { console.log("_emscripten_notify_mailbox_postmessage(", ...args); },
            "_emscripten_receive_on_main_thread_js": (...args) => { console.log("_emscripten_receive_on_main_thread_js(", ...args); },
            "_emscripten_thread_mailbox_await": (...args) => { console.log("_emscripten_thread_mailbox_await(", ...args); },
            "_emscripten_thread_set_strongref": (...args) => { console.log("_emscripten_thread_set_strongref(", ...args); },
            "_localtime_js": (time_low, time_high, tmPtr) => { },
            "_mmap_js": (...args) => { console.log("_mmap_js(", ...args); },
            "_munmap_js": (...args) => { console.log("_munmap_js(", ...args); },
            "_tzset_js": (timezone, daylight, std_name, dst_name) => { },
            "abort": (...args) => { console.log("abort(", ...args); },
            "emscripten_check_blocking_allowed": (...args) => { console.log("emscripten_check_blocking_allowed(", ...args); },
            "emscripten_date_now": () => Date.now(),
            "emscripten_err": (...args) => { console.log("emscripten_err(", ...args); },
            "emscripten_exit_with_live_runtime": (...args) => { console.log("emscripten_exit_with_live_runtime(", ...args); },
            "emscripten_get_now": (...args) => { console.log("emscripten_get_now(", ...args); },
            "emscripten_num_logical_cores": (...args) => { console.log("emscripten_num_logical_cores(", ...args); },
            "emscripten_resize_heap": (...args) => { console.log("emscripten_resize_heap(", ...args); },
            "environ_get": (a, b) => { return this.#environ_get(a, b); },
            "environ_sizes_get": (a, b) => { return this.#environ_sizes_get(a, b); },
            "exit": (...args) => { console.log("exit(", ...args); },
            "fd_close": (...args) => { console.log("fd_close(", ...args); },
            "fd_read": (fd, iov, iovcnt, pnum) => { this.#fd_read(fd, iov, iovcnt, pnum); },
            "fd_seek": (...args) => { console.log("fd_seek(", ...args); },
            "fd_write": (fd, iov, iovcnt, pnum) => { this.#fd_write(fd, iov, iovcnt, pnum); },
            "getentropy": (...args) => { console.log("getentropy(", ...args); },
            "strftime_l": (s, maxsize, format, tm, loc) => { this.heap8.set('TBD', s); return 4; },
        };
        return {
            env: imports,
            'wasi_snapshot_preview1': imports,
        }
    }
}

